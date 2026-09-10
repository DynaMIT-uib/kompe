"""Backend-aware linear-operator wrapper."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from kompe.math.backend import (
    _is_jax_tracer,
    block_until_ready,
    get_array_module,
    get_backend,
    immutable_array,
    readonly_numpy_array,
    to_numpy,
)

_NORMAL_MATRIX_WORK_BYTES = 64 * 1024**2
_WEIGHTED_PRODUCT_WORK_BYTES = 512 * 1024**2

MatrixShape: TypeAlias = tuple[int, int]
VectorizedMapFunc: TypeAlias = Callable[[Any], Any]
ArrayBackend: TypeAlias = Literal["numpy", "jax"]


@dataclass(frozen=True, init=False, eq=False)
class LinearMap:
    """Backend-agnostic linear map between shaped scientific arrays.

    Construct a matrix-free map from its forward and adjoint actions. Optional
    block, dense, sparse, diagonal, and normal-matrix functions preserve useful
    structure. ``map(values)`` retains domain/codomain and trailing batch
    axes. ``map @ values``, ``matvec``, and ``matmat`` use flat linear algebra;
    ``map @ other_map`` composes maps.

    A map has a fixed mathematical action. Array factories may borrow their
    numerical data; do not mutate those data while the map or its cached fits
    are in use. Construct a new map for a changed operator.
    """

    shape: MatrixShape
    dtype: Any
    _matvec: VectorizedMapFunc = field(repr=False)
    _rmatvec: VectorizedMapFunc = field(repr=False)
    _matmat: VectorizedMapFunc | None = field(default=None, repr=False)
    _rmatmat: VectorizedMapFunc | None = field(default=None, repr=False)
    _dense_array_func: Callable[[Any], Any] | None = field(default=None, repr=False)
    _sparse_matrix_func: Callable[[], Any] | None = field(default=None, repr=False)
    _sparse_cache: Any = field(default=None, repr=False)
    _diagonal_array_func: Callable[[Any], Any] | None = field(default=None, repr=False)
    _normal_matrix_func: Callable | None = field(default=None, repr=False)
    _normal_matrix_diag: Callable[..., np.ndarray] | None = field(default=None, repr=False)
    _backend_operands: tuple[Any, ...] = field(default=(), repr=False)
    _is_identity: bool = field(default=False, repr=False)
    _einsum_map: Any = field(default=None, repr=False, compare=False)
    _dense_tensor: Any = field(default=None, repr=False, compare=False)
    output_shape: tuple[int, ...] | None = None
    input_shape: tuple[int, ...] | None = None
    _dense_cache: dict[Any, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __init__(
        self,
        shape: MatrixShape,
        dtype: Any,
        matvec: VectorizedMapFunc,
        rmatvec: VectorizedMapFunc,
        matmat: VectorizedMapFunc | None = None,
        rmatmat: VectorizedMapFunc | None = None,
        *,
        dense_array: Callable[[Any], Any] | None = None,
        sparse_matrix: Callable[[], Any] | None = None,
        diagonal: Callable[[Any], Any] | None = None,
        normal_matrix: Callable[[Any, Any], Any] | None = None,
        normal_matrix_diag: Callable[..., np.ndarray] | None = None,
        backend_operands: tuple[Any, ...] = (),
        output_shape: tuple[int, ...] | None = None,
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize a map from forward and adjoint vector operations.

        A ``normal_matrix(xp, row_scale)`` callback materializes
        A* diag(abs(row_scale)²) A without expanding A.
        A normal-diagonal callback accepts optional ``row_scale=None`` and
        computes diag(A* diag(abs(row_scale)²) A), or diag(A* A) when omitted.
        This retains cheap reductions through weighted operator compositions.
        """
        shape = tuple(int(dimension) for dimension in shape)
        output_shape, input_shape = _map_shapes(shape, input_shape, output_shape)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "_matvec", matvec)
        object.__setattr__(self, "_rmatvec", rmatvec)
        object.__setattr__(self, "_matmat", matmat)
        object.__setattr__(self, "_rmatmat", rmatmat)
        object.__setattr__(self, "_dense_array_func", dense_array)
        object.__setattr__(self, "_sparse_matrix_func", sparse_matrix)
        object.__setattr__(self, "_sparse_cache", None)
        object.__setattr__(self, "_diagonal_array_func", diagonal)
        object.__setattr__(self, "_normal_matrix_func", normal_matrix)
        object.__setattr__(self, "_normal_matrix_diag", normal_matrix_diag)
        object.__setattr__(self, "_backend_operands", tuple(backend_operands))
        object.__setattr__(self, "_is_identity", False)
        object.__setattr__(self, "_einsum_map", None)
        object.__setattr__(self, "_dense_tensor", None)
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "_dense_cache", {})

    @property
    def ndim(self) -> int:
        """Dimensionality of the linear map."""
        return 2

    @property
    def backend_operands(self) -> tuple[Any, ...]:
        """Closed-over operands used for backend selection."""
        return self._backend_operands

    def array_module(self, *operands: Any) -> Any:
        """Return the array module implied by operands and this map."""
        return get_array_module(*operands, *self._backend_operands)

    def clear_dense_cache(self) -> None:
        """Discard derived dense materializations of this map."""
        self._dense_cache.clear()

    @property
    def materialized_matrix(self):
        """Return an existing dense matrix, or None without constructing one.

        Prefer the active backend's copy, then any existing representation.
        The result stays on its own backend. Declared diagonal maps should
        still use ``diagonal()`` for their vector-backed numerical action.
        """
        dense = self._dense_cache.get(self.array_module())
        if dense is None:
            dense = next(iter(self._dense_cache.values()), self._dense_tensor)
        return None if dense is None else dense.reshape(self.shape)

    @property
    def is_diagonal(self) -> bool:
        """Whether this map has an exact diagonal representation."""
        return self._diagonal_array_func is not None

    @property
    def is_sparse(self) -> bool:
        """Whether exact sparse structure is known, including diagonal maps."""
        return self.is_diagonal or self._sparse_matrix_func is not None

    def to_sparse_matrix(self):
        """Return a SciPy CSR matrix from declared sparse structure.

        This explicit CPU boundary never probes or densifies a map. Sparse
        algebra is evaluated lazily and retained for subsequent factorizations.
        Diagonal vectors are transferred only when this method is requested.
        """
        if self._sparse_cache is None:
            if self.is_diagonal:
                matrix = scipy.sparse.diags(self.diagonal(backend="numpy"), format="csr")
            elif self._sparse_matrix_func is not None:
                matrix = self._sparse_matrix_func().tocsr()
            else:
                raise ValueError("This map has no declared sparse representation.")
            object.__setattr__(self, "_sparse_cache", matrix)
        return self._sparse_cache

    def __call__(self, values: Any) -> Any:
        """Map ``input_shape + batch_shape`` to ``output_shape + batch_shape``.

        Domain axes come first; trailing axes index independent fields.
        Batches use the block action without materializing the map.
        """
        xp = self.array_module(values)
        values = xp.asarray(values)
        input_ndim = len(self.input_shape)
        if values.shape[:input_ndim] != self.input_shape:
            raise ValueError(
                f"Values must start with input_shape {self.input_shape}; got {values.shape}."
            )
        batch_shape = values.shape[input_ndim:]
        if not batch_shape:
            return self.matvec(values.reshape(self.shape[1])).reshape(self.output_shape)
        block = values.reshape(self.shape[1], math.prod(batch_shape))
        return self.matmat(block).reshape(self.output_shape + batch_shape)

    def matvec(self, x: Any) -> Any:
        """Apply this map to one flattened vector."""
        xp = self.array_module(x)
        dense = self._dense_cache.get(xp)
        if dense is not None and self._diagonal_array_func is None:
            x_arr = xp.asarray(x).reshape(self.shape[1])
            return dense @ x_arr
        return self._matvec(x)

    def rmatvec(self, y: Any) -> Any:
        """Apply the adjoint map to one flattened vector."""
        xp = self.array_module(y)
        dense = self._dense_cache.get(xp)
        if dense is not None and self._diagonal_array_func is None:
            y_arr = xp.asarray(y).reshape(self.shape[0])
            return dense.T.conj() @ y_arr
        return self._rmatvec(y)

    def matmat(self, x_block: Any) -> Any:
        """Apply to columns, or retain a single vector's one-dimensional shape."""
        xp = self.array_module(x_block)
        x_arr = _dense_array_candidate(x_block)
        if x_arr.ndim == 1:
            return self.matvec(x_arr)
        if x_arr.ndim != 2 or x_arr.shape[0] != self.shape[1]:
            raise ValueError(f"Column blocks must have shape ({self.shape[1]}, n_columns).")
        if 0 in self.shape or x_arr.shape[1] == 0:
            return xp.zeros(
                (self.shape[0], x_arr.shape[1]), dtype=xp.result_type(self.dtype, x_arr.dtype)
            )
        dense = self._dense_cache.get(xp)
        if dense is not None and self._diagonal_array_func is None:
            return dense @ xp.asarray(x_arr)
        if self._matmat is not None:
            return self._matmat(x_arr)
        outputs = [self.matvec(x_arr[:, i]) for i in range(x_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        """Apply the adjoint to columns, or retain a single vector's shape."""
        xp = self.array_module(y_block)
        y_arr = _dense_array_candidate(y_block)
        if y_arr.ndim == 1:
            return self.rmatvec(y_arr)
        if y_arr.ndim != 2 or y_arr.shape[0] != self.shape[0]:
            raise ValueError(f"Column blocks must have shape ({self.shape[0]}, n_columns).")
        if 0 in self.shape or y_arr.shape[1] == 0:
            return xp.zeros(
                (self.shape[1], y_arr.shape[1]), dtype=xp.result_type(self.dtype, y_arr.dtype)
            )
        dense = self._dense_cache.get(xp)
        if dense is not None and self._diagonal_array_func is None:
            adjoint = dense.T.conj()
            return adjoint @ xp.asarray(y_arr)
        if self._rmatmat is not None:
            return self._rmatmat(y_arr)
        outputs = [self.rmatvec(y_arr[:, i]) for i in range(y_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def _dense_array(self, xp: Any = None) -> Any:
        """Materialize this map as a dense 2-D matrix on ``xp``."""
        xp = self.array_module() if xp is None else xp
        cached = self._dense_cache.get(xp)
        if cached is not None:
            return cached

        if self._dense_array_func is not None:
            dense = self._dense_array_func(xp)
        else:
            eye_dtype = np.result_type(self.dtype, np.float64)
            if self.shape[0] < self.shape[1]:
                output_identity = xp.eye(self.shape[0], dtype=eye_dtype)
                adjoint = self.rmatmat(output_identity)
                dense = adjoint.T.conj()
            else:
                input_identity = xp.eye(self.shape[1], dtype=eye_dtype)
                dense = self.matmat(input_identity)
            dense = xp.asarray(dense)
        # Traced materialization belongs to the compiled function,
        # not this reusable Python object's eager cache.
        if not _is_jax_tracer(dense):
            self._dense_cache[xp] = dense
        return dense

    def to_matrix(self, *, backend: ArrayBackend | None = None) -> Any:
        """Materialize this map as an explicit flat 2-D matrix."""
        xp = get_array_module(*self._backend_operands, backend=backend)
        return block_until_ready(self._dense_array(xp))

    def to_array(self, *, backend: ArrayBackend | None = None) -> Any:
        """Materialize this map with its shaped domain and codomain."""
        xp = get_array_module(*self._backend_operands, backend=backend)
        dense = self._dense_array(xp)
        return block_until_ready(xp.reshape(dense, self.output_shape + self.input_shape))

    def adjoint(self) -> LinearMap:
        """Return the conjugate-transpose linear map."""
        diagonal_func = None
        if self.is_diagonal:

            def diagonal_func(xp: Any) -> Any:
                return xp.conjugate(self._diagonal_array(xp))

        def dense_array(xp: Any) -> Any:
            dense = self._dense_array(xp)
            return dense.T.conj()

        return LinearMap(
            shape=(self.shape[1], self.shape[0]),
            dtype=self.dtype,
            matvec=self.rmatvec,
            rmatvec=self.matvec,
            matmat=self.rmatmat,
            rmatmat=self.matmat,
            dense_array=dense_array,
            diagonal=diagonal_func,
            sparse_matrix=(lambda: self.to_sparse_matrix().T.conjugate())
            if self.is_sparse
            else None,
            backend_operands=self.backend_operands,
            input_shape=self.output_shape,
            output_shape=self.input_shape,
        )

    def diagonal(self, *, backend: ArrayBackend | None = None) -> Any:
        """Return scale values from a known diagonal representation.

        This never materializes a dense matrix to discover structure.
        """
        xp = get_array_module(*self._backend_operands, backend=backend)
        return block_until_ready(self._diagonal_array(xp))

    def _diagonal_array(self, xp: Any = None) -> Any:
        """Return exact diagonal scale values on ``xp``."""
        xp = self.array_module() if xp is None else xp
        if self._diagonal_array_func is not None:
            return self._diagonal_array_func(xp)
        raise ValueError(
            "This map has no known diagonal representation. "
            "Use diagonal_linear_map to declare diagonal structure."
        )

    def normal_operator(self, row_scale=None):
        """Return A* diag(abs(row_scale)²) A, retaining operator structure.

        ``row_scale`` contains one scale per flat output row (None means
        unit weights). Application is matrix-free; explicit materialization
        reuses dense, sparse, diagonal, or declared normal-product structure.
        The result maps ``input_shape`` to itself. No solver is selected.
        """
        if row_scale is not None:
            row_scale = immutable_array(row_scale).reshape(-1)
            if row_scale.shape != (self.shape[0],):
                raise ValueError("row_scale must contain one scale per output row.")
        if self.is_diagonal:
            xp = self.array_module(row_scale)
            diagonal = self._diagonal_array(xp)
            if row_scale is not None:
                diagonal = diagonal * xp.asarray(row_scale)
            return diagonal_linear_map(
                xp.abs(diagonal) ** 2, input_shape=self.input_shape, output_shape=self.input_shape
            )

        xp = self.array_module(row_scale)
        squared_scale = None if row_scale is None else xp.abs(xp.asarray(row_scale)) ** 2

        def matvec(x):
            values = self.matvec(x)
            if row_scale is not None:
                xp = self.array_module(values, row_scale)
                values = xp.asarray(squared_scale) * values
            return self.rmatvec(values)

        def matmat(x):
            values = self.matmat(x)
            if row_scale is not None:
                xp = self.array_module(values, row_scale)
                values = xp.asarray(squared_scale)[:, None] * values
            return self.rmatmat(values)

        def sparse_matrix():
            matrix = self.to_sparse_matrix()
            if row_scale is not None:
                matrix = matrix.multiply(to_numpy(row_scale)[:, None]).tocsr()
            return matrix.T.conj() @ matrix

        return LinearMap(
            shape=(self.shape[1], self.shape[1]),
            dtype=np.result_type(self.dtype, getattr(row_scale, "dtype", self.dtype)),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            rmatmat=matmat,
            dense_array=lambda xp: self._normal_matrix(xp, row_scale),
            sparse_matrix=sparse_matrix if self.is_sparse else None,
            backend_operands=self.backend_operands + (() if row_scale is None else (row_scale,)),
            input_shape=self.input_shape,
            output_shape=self.input_shape,
        )

    def _normal_matrix(self, xp, row_scale=None):
        """Materialize a normal product without retaining a second cache."""
        matrix = self.materialized_matrix
        if matrix is not None:
            matrix = self._dense_array(xp)
            if row_scale is None:
                return matrix.T.conj() @ matrix
            scale = xp.asarray(to_numpy(row_scale) if xp is np else row_scale)
            return _weighted_cross_product(matrix, matrix, xp.abs(scale) ** 2)
        if self._normal_matrix_func is not None:
            result = self._normal_matrix_func(xp, row_scale)
            return xp.asarray(to_numpy(result) if xp is np else result)
        if self.is_sparse:
            return xp.asarray(self.normal_operator(row_scale).to_sparse_matrix().toarray())

        # Bound temporary directions and sampled columns. Unlike A* A via
        # a dense A, this also works when the sampled field is very large.
        n = self.shape[1]
        dtype = np.result_type(self.dtype, getattr(row_scale, "dtype", self.dtype))
        normal = xp.zeros((n, n), dtype=dtype)
        bytes_per_column = max(1, (2 * n + self.shape[0]) * np.dtype(dtype).itemsize)
        block_size = max(1, _NORMAL_MATRIX_WORK_BYTES // bytes_per_column)
        operator = self.normal_operator(row_scale)
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            directions = xp.eye(n, stop - start, k=-start, dtype=dtype)
            columns = operator.matmat(directions)
            if xp is np:
                normal[:, start:stop] = to_numpy(columns)
            else:
                normal = normal.at[:, start:stop].set(columns)
        return normal

    def normal_matrix_diag(self, row_scale=None) -> np.ndarray:
        """Return the CPU normal diagonal, optionally after scaling flat rows.

        Keep diagonal vectors compact; otherwise reuse an existing matrix
        on its backend, transferring only the resulting diagonal. Without
        materialized values, use a structured formula or bounded column probes.
        With row_scale w, the result is diag(A* diag(abs(w)²) A).
        """
        if row_scale is not None and np.shape(row_scale) != (self.shape[0],):
            raise ValueError("row_scale must contain one value per flat output row.")
        if self._diagonal_array_func is not None:
            values = np.asarray(self.diagonal(backend="numpy"))
            if row_scale is not None:
                values = values * to_numpy(row_scale)
            return np.abs(values) ** 2
        dense = self.materialized_matrix
        if dense is not None:
            xp = _runtime_array_module(dense)
            if row_scale is not None:
                dense = xp.asarray(row_scale)[:, None] * dense
            return to_numpy(xp.sum(xp.abs(dense) ** 2, axis=0))
        if self._normal_matrix_diag is not None:
            diagonal = (
                self._normal_matrix_diag()
                if row_scale is None
                else self._normal_matrix_diag(row_scale=row_scale)
            )
            return np.asarray(diagonal).real
        return _normal_matrix_diag_from_matmat(self.shape, self.dtype, self.matmat, row_scale)

    def __matmul__(self, other: Any) -> Any:
        """Apply to arrays or compose with another operator."""
        looks_like_operator = isinstance(other, (LinearMap, ScipyLinearOperator)) or hasattr(
            other, "matvec"
        )
        if not scipy.sparse.issparse(other) and not looks_like_operator:
            arr = _dense_array_candidate(other)
            if arr.ndim == 1:
                return self.matvec(arr)
            if arr.ndim == 2:
                return self.matmat(arr)

        return self._compose(as_linear_map(other))

    def _compose(self, other_map: LinearMap) -> LinearMap:
        """Compose this map with a compatible map on its right."""
        if self.shape[1] != other_map.shape[0]:
            raise ValueError(
                f"Dimension mismatch for composition: {self.shape} @ {other_map.shape}"
            )
        if self._is_identity:
            return as_linear_map(
                other_map, input_shape=other_map.input_shape, output_shape=self.output_shape
            )
        if other_map._is_identity:
            return as_linear_map(
                self, input_shape=other_map.input_shape, output_shape=self.output_shape
            )
        from kompe.math.einsum import fuse_linear_maps

        fused = fuse_linear_maps(self, other_map)
        return self._composed_linear_map(other_map) if fused is None else fused

    def _composed_linear_map(self, other_map: LinearMap) -> LinearMap:
        """Build the lazy fallback representation of a composition."""
        self_is_diagonal = self._diagonal_array_func is not None
        other_is_diagonal = other_map._diagonal_array_func is not None

        def matvec(x: Any) -> Any:
            return self.matvec(other_map.matvec(x))

        def rmatvec(y: Any) -> Any:
            return other_map.rmatvec(self.rmatvec(y))

        def matmat(x: Any) -> Any:
            return self.matmat(other_map.matmat(x))

        def rmatmat(y: Any) -> Any:
            return other_map.rmatmat(self.rmatmat(y))

        dense_array = None
        if self_is_diagonal or other_is_diagonal:

            def dense_array(xp: Any) -> Any:
                # Scale rows or columns without expanding diagonal factors.
                if self_is_diagonal and other_is_diagonal:
                    return xp.diag(self._diagonal_array(xp) * other_map._diagonal_array(xp))
                if self_is_diagonal:
                    return self._diagonal_array(xp)[:, None] * other_map._dense_array(xp)
                return self._dense_array(xp) * other_map._diagonal_array(xp)[None, :]

        dtype = np.promote_types(self.dtype, other_map.dtype)

        diagonal_array = None
        if self_is_diagonal and other_is_diagonal:

            def diagonal_array(xp: Any) -> Any:
                return self._diagonal_array(xp) * other_map._diagonal_array(xp)

        normal_matrix_diag = None
        if self_is_diagonal:

            def normal_matrix_diag(row_scale=None) -> np.ndarray:
                diagonal = self.diagonal()
                if row_scale is not None:
                    xp = get_array_module(diagonal, row_scale)
                    diagonal = xp.asarray(diagonal) * xp.asarray(row_scale)
                return other_map.normal_matrix_diag(row_scale=diagonal)

        elif other_is_diagonal:

            def normal_matrix_diag(row_scale=None) -> np.ndarray:
                diagonal = np.asarray(other_map.diagonal(backend="numpy"))
                return np.abs(diagonal) ** 2 * self.normal_matrix_diag(row_scale=row_scale)

        def normal_matrix(xp, row_scale):
            if self_is_diagonal:
                scale = self._diagonal_array(xp)
                if row_scale is not None:
                    scale = scale * xp.asarray(row_scale)
                return other_map._normal_matrix(xp, scale)
            normal = self._normal_matrix(xp, row_scale)
            if other_is_diagonal:
                scale = other_map._diagonal_array(xp)
                return scale.conj()[:, None] * normal * scale[None, :]
            # Z* N Z applies an arbitrary coordinate restriction without
            # expanding Z (notably an orthonormal gauge-nullspace map).
            left = other_map.rmatmat(normal)
            return other_map.rmatmat(left.T.conj()).T.conj()

        return LinearMap(
            shape=(self.shape[0], other_map.shape[1]),
            dtype=dtype,
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dense_array=dense_array,
            diagonal=diagonal_array,
            normal_matrix_diag=normal_matrix_diag,
            normal_matrix=normal_matrix
            if self_is_diagonal or other_is_diagonal or self._normal_matrix_func is not None
            else None,
            backend_operands=self._backend_operands + other_map._backend_operands,
            sparse_matrix=(lambda: self.to_sparse_matrix() @ other_map.to_sparse_matrix())
            if self.is_sparse and other_map.is_sparse
            else None,
            output_shape=self.output_shape,
            input_shape=other_map.input_shape,
        )

    def __add__(self, other: Any) -> LinearMap:
        """Add two linear maps with identical shaped domains."""
        other_map = as_linear_map(other)
        if self.shape != other_map.shape:
            raise ValueError(f"Shape mismatch for addition: {self.shape} + {other_map.shape}")
        if (
            self.output_shape != other_map.output_shape
            or self.input_shape != other_map.input_shape
        ):
            raise ValueError(
                "Shape metadata mismatch for addition: "
                f"{self.output_shape} <- {self.input_shape} and "
                f"{other_map.output_shape} <- {other_map.input_shape}"
            )

        def matvec(x: Any) -> Any:
            return self.matvec(x) + other_map.matvec(x)

        def rmatvec(y: Any) -> Any:
            return self.rmatvec(y) + other_map.rmatvec(y)

        def matmat(x: Any) -> Any:
            return self.matmat(x) + other_map.matmat(x)

        def rmatmat(y: Any) -> Any:
            return self.rmatmat(y) + other_map.rmatmat(y)

        def dense_array(xp: Any) -> Any:
            return xp.asarray(self._dense_array(xp)) + xp.asarray(other_map._dense_array(xp))

        dtype = np.promote_types(self.dtype, other_map.dtype)

        return LinearMap(
            shape=self.shape,
            dtype=dtype,
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dense_array=dense_array,
            diagonal=(lambda xp: self._diagonal_array(xp) + other_map._diagonal_array(xp))
            if self.is_diagonal and other_map.is_diagonal
            else None,
            backend_operands=self._backend_operands + other_map._backend_operands,
            sparse_matrix=(lambda: self.to_sparse_matrix() + other_map.to_sparse_matrix())
            if self.is_sparse and other_map.is_sparse
            else None,
            output_shape=self.output_shape,
            input_shape=self.input_shape,
        )

    def __radd__(self, other: Any) -> LinearMap:
        """Add two linear maps with identical shaped domains."""
        if np.isscalar(other) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: Any) -> LinearMap:
        """Subtract another linear map."""
        return self.__add__(-as_linear_map(other))

    def __mul__(self, other: Any) -> LinearMap:
        """Scale this linear map."""
        if not np.isscalar(other):
            return NotImplemented
        scalar = other
        if not self.is_diagonal:
            dense = self._dense_cache.get(self.array_module())
            if dense is None:
                dense = self._dense_tensor
            if dense is not None:
                return as_linear_map(
                    dense * scalar, input_shape=self.input_shape, output_shape=self.output_shape
                )
        if (
            self._einsum_map is not None
            and self._einsum_map.output_shape == self.output_shape
            and self._einsum_map.input_shape == self.input_shape
        ):
            from kompe.math.einsum import scale_einsum_map

            return scale_einsum_map(self._einsum_map, scalar).to_linear_map()

        def matvec(x: Any) -> Any:
            return self.matvec(x) * scalar

        def rmatvec(y: Any) -> Any:
            return self.rmatvec(y) * scalar.conjugate()

        def matmat(x: Any) -> Any:
            return self.matmat(x) * scalar

        def rmatmat(y: Any) -> Any:
            return self.rmatmat(y) * scalar.conjugate()

        def dense_array(xp: Any) -> Any:
            return self._dense_array(xp) * scalar

        def normal_matrix_diag(row_scale=None) -> np.ndarray:
            return np.abs(scalar) ** 2 * self.normal_matrix_diag(row_scale=row_scale)

        def diagonal_array(xp: Any) -> Any:
            return self._diagonal_array(xp) * scalar

        return LinearMap(
            shape=self.shape,
            dtype=np.result_type(self.dtype, scalar),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dense_array=dense_array,
            diagonal=(diagonal_array if self._diagonal_array_func is not None else None),
            normal_matrix_diag=normal_matrix_diag,
            normal_matrix=lambda xp, row_scale: (
                abs(scalar) ** 2 * self._normal_matrix(xp, row_scale)
            ),
            backend_operands=self._backend_operands,
            sparse_matrix=(lambda: self.to_sparse_matrix() * scalar) if self.is_sparse else None,
            output_shape=self.output_shape,
            input_shape=self.input_shape,
        )

    def __rmul__(self, other: Any) -> LinearMap:
        """Scale this linear map."""
        return self.__mul__(other)

    def __neg__(self) -> LinearMap:
        """Negate this linear map."""
        return -1.0 * self

    def as_linear_operator(self) -> ScipyLinearOperator:
        """Return a SciPy ``LinearOperator`` view of this map."""

        def matvec(vec: np.ndarray) -> np.ndarray:
            return np.asarray(self.matvec(vec))

        def rmatvec(vec: np.ndarray) -> np.ndarray:
            return np.asarray(self.rmatvec(vec))

        def matmat(block: np.ndarray) -> np.ndarray:
            return np.asarray(self.matmat(block))

        def rmatmat(block: np.ndarray) -> np.ndarray:
            return np.asarray(self.rmatmat(block))

        return ScipyLinearOperator(
            self.shape,
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=self.dtype,
        )


def _runtime_array_module(*values: Any) -> Any:
    """Select JAX only when an operand is already a JAX array."""
    if any("jax" in type(value).__module__ for value in values):
        return get_array_module(*values)
    return np


def _normal_matrix_diag_from_matmat(
    shape: MatrixShape, dtype: Any, matmat: Callable[[Any], Any], row_scale=None
) -> np.ndarray:
    """Compute ``diag(A* A)`` from bounded identity blocks."""
    n_cols = shape[1]
    work_dtype = np.result_type(dtype, np.float64)
    diag_dtype = np.empty((), dtype=work_dtype).real.dtype
    diag = np.zeros(n_cols, dtype=diag_dtype)
    block_size = min(32, max(1, n_cols))
    block = np.zeros((n_cols, block_size), dtype=work_dtype)
    for start in range(0, n_cols, block_size):
        stop = min(n_cols, start + block_size)
        cols = stop - start
        block[:, :cols] = 0
        block[start:stop, :cols] = np.eye(cols, dtype=work_dtype)
        res = matmat(block[:, :cols])
        xp = _runtime_array_module(res)
        if row_scale is not None:
            res = res * xp.asarray(row_scale)[:, None]
        diag[start:stop] = to_numpy(xp.sum(xp.abs(res) ** 2, axis=0)).real
    return diag


def _dense_array_candidate(value: Any) -> Any:
    """Return dense input without materializing backend arrays."""
    if (
        getattr(value, "shape", None) is not None
        and getattr(value, "ndim", None) is not None
        and getattr(value, "dtype", None) is not None
    ):
        return value
    return np.asarray(value)


def _map_shapes(
    shape: MatrixShape,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return shape metadata compatible with flat dimensions."""
    in_shape = (shape[1],) if input_shape is None else tuple(input_shape)
    out_shape = (shape[0],) if output_shape is None else tuple(output_shape)
    if math.prod(in_shape) != shape[1]:
        raise ValueError(f"Input shape {in_shape} incompatible with operator {shape}.")
    if math.prod(out_shape) != shape[0]:
        raise ValueError(f"Output shape {out_shape} incompatible with operator {shape}.")
    return out_shape, in_shape


def _linear_map_from_dense(
    matrix: Any,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> LinearMap:
    mat_array = _dense_array_candidate(matrix)
    if mat_array.ndim != 2:
        raise ValueError("Dense operators must be 2-D arrays.")
    shape = tuple(int(dim) for dim in mat_array.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = mat_array.dtype

    def matvec(vec: Any) -> Any:
        xp = _runtime_array_module(mat_array, vec)
        mat_arr = xp.asarray(mat_array)
        vec_arr = xp.asarray(vec).reshape(shape[1])
        return xp.matmul(mat_arr, vec_arr)

    def rmatvec(vec: Any) -> Any:
        xp = _runtime_array_module(mat_array, vec)
        mat_arr = xp.asarray(mat_array)
        vec_arr = xp.asarray(vec).reshape(shape[0])
        return mat_arr.T.conj() @ vec_arr

    def matmat(block: Any) -> Any:
        xp = _runtime_array_module(mat_array, block)
        mat_arr = xp.asarray(mat_array)
        block_arr = xp.asarray(block).reshape(shape[1], -1)
        return xp.matmul(mat_arr, block_arr)

    def rmatmat(block: Any) -> Any:
        xp = _runtime_array_module(mat_array, block)
        mat_arr = xp.asarray(mat_array)
        block_arr = xp.asarray(block).reshape(shape[0], -1)
        adjoint = mat_arr.T.conj()
        return xp.matmul(adjoint, block_arr)

    def dense_array(xp: Any) -> Any:
        return xp.asarray(mat_array)

    linear_map = LinearMap(
        shape=shape,
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dense_array=dense_array,
        backend_operands=(mat_array,),
        output_shape=out_shape,
        input_shape=in_shape,
    )
    object.__setattr__(linear_map, "_dense_tensor", mat_array.reshape(out_shape + in_shape))
    return linear_map


def diagonal_linear_map(
    diag_values: Any,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> LinearMap:
    """Return a map backed by a diagonal vector."""
    diag_array = _dense_array_candidate(diag_values).reshape(-1)
    size = int(diag_array.size)
    out_shape, in_shape = _map_shapes((size, size), input_shape, output_shape)
    dtype = diag_array.dtype

    def matvec(vec: Any) -> Any:
        xp = _runtime_array_module(diag_array, vec)
        diag_arr = xp.asarray(diag_array)
        vec_arr = xp.asarray(vec).reshape(size)
        return diag_arr * vec_arr

    def rmatvec(vec: Any) -> Any:
        xp = _runtime_array_module(diag_array, vec)
        diag_arr = xp.asarray(diag_array)
        vec_arr = xp.asarray(vec).reshape(size)
        return xp.conjugate(diag_arr) * vec_arr

    def matmat(block: Any) -> Any:
        xp = _runtime_array_module(diag_array, block)
        diag_arr = xp.asarray(diag_array).reshape(size, 1)
        block_arr = xp.asarray(block).reshape(size, -1)
        return diag_arr * block_arr

    def rmatmat(block: Any) -> Any:
        xp = _runtime_array_module(diag_array, block)
        diag_arr = xp.asarray(diag_array).reshape(size, 1)
        block_arr = xp.asarray(block).reshape(size, -1)
        return xp.conjugate(diag_arr) * block_arr

    def dense_array(xp: Any) -> Any:
        return xp.diag(xp.asarray(diag_array))

    def diagonal_array(xp: Any) -> Any:
        return xp.asarray(diag_array).reshape(size)

    return LinearMap(
        shape=(size, size),
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dense_array=dense_array,
        diagonal=diagonal_array,
        backend_operands=(diag_array,),
        output_shape=out_shape,
        input_shape=in_shape,
    )


def _normalized_constraint_rows(rows):
    """Remove arbitrary row units from homogeneous constraints on the CPU."""
    if rows.shape[0] == 0:
        return rows
    if scipy.sparse.issparse(rows):
        rows = rows.astype(np.result_type(rows.dtype, 0.0), copy=True).tocsr()
        scale = np.asarray(abs(rows).max(axis=1).toarray()).reshape(-1)
        row_indices = np.repeat(np.arange(rows.shape[0]), np.diff(rows.indptr))
        rows.data /= np.where(scale > 0, scale, 1)[row_indices]
        norms = np.sqrt(np.asarray(abs(rows).power(2).sum(axis=1)).reshape(-1))
        rows.data /= np.where(norms > 0, norms, 1)[row_indices]
        return rows
    scale = np.max(np.abs(rows), axis=1, initial=0)
    scaled = rows / np.where(scale > 0, scale, 1)[:, None]
    norms = np.linalg.norm(scaled, axis=1)
    return scaled / np.where(norms > 0, norms, 1)[:, None]


def null_space_linear_map(constraint_matrix, *, output_shape=None) -> LinearMap:
    """Return an orthonormal basis for ``C x = 0`` without a dense nullspace.

    ``C`` must have independent rows and no more rows than columns.
    A CPU QR factorization retains one Householder vector per constraint;
    application uses the active array backend and costs O(n k) per RHS,
    where n is the coefficient count and k is the constraint count.
    The map takes n-k independent coordinates to the shaped full space.
    """
    constraints = np.asarray(constraint_matrix)
    if constraints.ndim != 2 or not np.all(np.isfinite(constraints)):
        raise ValueError("constraint_matrix must be a finite two-dimensional matrix.")
    k, n = constraints.shape
    if k > n:
        raise ValueError("constraint_matrix cannot have more rows than columns.")
    constraints = _normalized_constraint_rows(constraints)
    (qr, tau), triangular = scipy.linalg.qr(constraints.T.conj(), mode="raw")
    if k and np.linalg.matrix_rank(triangular) != k:
        raise ValueError("constraint_matrix must have independent rows.")
    vectors = np.tril(qr, -1) + np.eye(n, k, dtype=qr.dtype)
    reflectors = as_linear_map(vectors)

    def reflect(values, *, adjoint=False):
        xp = get_array_module(values)
        result = xp.asarray(values)
        device_vectors = reflectors.to_matrix(backend=get_backend(values))
        # Q = H_0 ... H_(k-1); its adjoint applies reflectors in reverse.
        for i in range(k) if adjoint else reversed(range(k)):
            vector = device_vectors[:, i : i + 1]
            scale = tau[i].conjugate() if adjoint else tau[i]
            result = result - scale * vector * (vector.T.conj() @ result)
        return result

    def matmat(values):
        xp = get_array_module(values)
        values = xp.asarray(values)
        return reflect(
            xp.concatenate([xp.zeros((k, values.shape[1]), dtype=values.dtype), values])
        )

    def rmatmat(values):
        return reflect(values, adjoint=True)[k:]

    def normal_matrix_diag(row_scale=None):
        # Z* Z = I. Preserve this structure when another coordinate
        # restriction needs column norms, without probing or materializing Z.
        if row_scale is None:
            return np.ones(n - k)
        return _normal_matrix_diag_from_matmat((n, n - k), vectors.dtype, matmat, row_scale)

    return LinearMap(
        shape=(n, n - k),
        dtype=vectors.dtype,
        matvec=lambda values: matmat(values.reshape(-1, 1)).reshape(-1),
        rmatvec=lambda values: rmatmat(values.reshape(-1, 1)).reshape(-1),
        matmat=matmat,
        rmatmat=rmatmat,
        normal_matrix_diag=normal_matrix_diag,
        input_shape=(n - k,),
        output_shape=output_shape,
    )


def identity_linear_map(shape: int | tuple[int, ...], *, dtype: Any = np.float64) -> LinearMap:
    """Return an identity map without storing an explicit diagonal."""
    value_shape = (int(shape),) if isinstance(shape, (int, np.integer)) else tuple(shape)
    size = int(math.prod(value_shape))
    dtype = np.dtype(dtype)

    def matvec(vec: Any) -> Any:
        xp = get_array_module(vec)
        return xp.asarray(vec).reshape(size)

    def matmat(block: Any) -> Any:
        xp = get_array_module(block)
        return xp.asarray(block).reshape(size, -1)

    def dense_array(xp: Any) -> Any:
        return xp.eye(size, dtype=dtype)

    def diagonal_array(xp: Any) -> Any:
        return xp.ones(size, dtype=dtype)

    identity = LinearMap(
        shape=(size, size),
        dtype=dtype,
        matvec=matvec,
        rmatvec=matvec,
        matmat=matmat,
        rmatmat=matmat,
        dense_array=dense_array,
        diagonal=diagonal_array,
        output_shape=value_shape,
        input_shape=value_shape,
    )
    object.__setattr__(identity, "_is_identity", True)
    return identity


def pointwise_component_map(array: Any) -> LinearMap:
    """Return a pointwise component map.

    ``array`` has shape
    ``(n_output_components, n_input_components, *points)`` and maps
    arrays shaped ``(n_input_components, *points)`` to
    ``(n_output_components, *points)``.
    """
    component_array = _dense_array_candidate(array)
    if component_array.ndim < 2:
        raise ValueError("pointwise array must have at least two component axes.")

    output_components = int(component_array.shape[0])
    input_components = int(component_array.shape[1])
    point_shape = tuple(int(dim) for dim in component_array.shape[2:])
    input_shape = (input_components,) + point_shape
    output_shape = (output_components,) + point_shape
    input_size = int(math.prod(input_shape))
    output_size = int(math.prod(output_shape))
    dtype = component_array.dtype

    def matvec(vec: Any) -> Any:
        xp = _runtime_array_module(component_array, vec)
        values = xp.asarray(vec).reshape(input_shape)
        result = xp.einsum("ab...,b...->a...", xp.asarray(component_array), values, optimize=True)
        return result.reshape(-1)

    def rmatvec(vec: Any) -> Any:
        xp = _runtime_array_module(component_array, vec)
        values = xp.asarray(vec).reshape(output_shape)
        result = xp.einsum(
            "ab...,a...->b...", xp.conjugate(xp.asarray(component_array)), values, optimize=True
        )
        return result.reshape(-1)

    def matmat(block: Any) -> Any:
        xp = _runtime_array_module(component_array, block)
        values = xp.asarray(block).reshape(input_shape + (-1,))
        result = xp.einsum(
            "ab...,b...h->a...h", xp.asarray(component_array), values, optimize=True
        )
        return result.reshape(output_size, -1)

    def rmatmat(block: Any) -> Any:
        xp = _runtime_array_module(component_array, block)
        values = xp.asarray(block).reshape(output_shape + (-1,))
        result = xp.einsum(
            "ab...,a...h->b...h", xp.conjugate(xp.asarray(component_array)), values, optimize=True
        )
        return result.reshape(input_size, -1)

    def normal_matrix_diag(row_scale=None) -> np.ndarray:
        values = to_numpy(component_array)
        if row_scale is not None:
            values = values * to_numpy(row_scale).reshape(output_shape)[:, None, ...]
        return np.sum(np.abs(values) ** 2, axis=0).reshape(-1)

    def dense_array(xp: Any) -> Any:
        point_size = int(math.prod(point_shape))
        matrix_values = xp.asarray(component_array).reshape(
            output_components, input_components, point_size
        )
        dense = xp.zeros((output_size, input_size), dtype=dtype)
        point_indices = xp.arange(point_size)
        for output_component in range(output_components):
            output_rows = output_component * point_size + point_indices
            for input_component in range(input_components):
                input_cols = input_component * point_size + point_indices
                values = matrix_values[output_component, input_component]
                if hasattr(dense, "at"):
                    dense = dense.at[output_rows, input_cols].set(values)
                else:
                    dense[output_rows, input_cols] = values
        return dense

    return LinearMap(
        shape=(output_size, input_size),
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dense_array=dense_array,
        normal_matrix_diag=normal_matrix_diag,
        backend_operands=(component_array,),
        input_shape=input_shape,
        output_shape=output_shape,
    )


def _normalize_take_selection(input_shape, indices, axis):
    """Return validated shapes, axis, and integer selection indices."""
    input_shape = tuple(operator.index(dim) for dim in input_shape)
    if not input_shape:
        raise ValueError("input_shape must have at least one axis.")
    axis = operator.index(axis)
    if axis < 0:
        axis += len(input_shape)
    if axis < 0 or axis >= len(input_shape):
        raise ValueError(f"axis {axis} is outside input_shape {input_shape}.")

    index_array = np.asarray(indices)
    if index_array.dtype == np.bool_:
        if index_array.shape != (input_shape[axis],):
            raise ValueError("Boolean indices must match the selected input axis length.")
        index_array = np.flatnonzero(index_array)
    if index_array.ndim > 1:
        raise ValueError("indices must be a scalar, a one-dimensional array, or an axis mask.")
    if index_array.size and not np.issubdtype(index_array.dtype, np.integer):
        raise TypeError("indices must contain integers.")
    if np.any(index_array < 0) or np.any(index_array >= input_shape[axis]):
        raise IndexError("take_linear_map indices are outside input_shape.")

    output_shape = list(input_shape)
    if index_array.ndim == 0:
        selection = int(index_array)
        del output_shape[axis]
    else:
        selection = readonly_numpy_array(index_array, dtype=np.intp)
        output_shape[axis] = int(index_array.size)
    output_shape = tuple(output_shape)
    return input_shape, output_shape, selection, axis


def take_linear_map(
    input_shape: tuple[int, ...], indices: Any, *, axis: int = -1, dtype: Any = np.float64
) -> LinearMap:
    """Select values along one shaped axis without a dense matrix.

    A scalar index removes the selected axis; an index array retains it.
    Scalar selections use basic slicing. Array selections may repeat
    indices, in which case the adjoint adds their contributions.
    """
    input_shape, output_shape, selection, axis = _normalize_take_selection(
        input_shape, indices, axis
    )
    input_size = int(math.prod(input_shape))
    output_size = int(math.prod(output_shape))
    dtype = np.dtype(dtype)
    scalar_selection = isinstance(selection, int)
    prefix = (slice(None),) * axis
    scalar_indexer = prefix + (selection,) if scalar_selection else None

    def _indexer(xp: Any):
        return scalar_indexer if scalar_selection else prefix + (xp.asarray(selection),)

    def matvec(vec: Any) -> Any:
        xp = get_array_module(vec)
        values = xp.asarray(vec).reshape(input_shape)
        return values[_indexer(xp)].reshape(-1)

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(vec)
        values = xp.asarray(vec).reshape(output_shape)
        result = xp.zeros(input_shape, dtype=values.dtype)
        indexer = _indexer(xp)
        if hasattr(result, "at"):
            update = result.at[indexer]
            return (update.set(values) if scalar_selection else update.add(values)).reshape(-1)
        if scalar_selection:
            result[indexer] = values
        else:
            np.add.at(result, indexer, values)
        return result.reshape(-1)

    def matmat(block: Any) -> Any:
        xp = get_array_module(block)
        values = xp.asarray(block).reshape(input_shape + (-1,))
        selected = values[_indexer(xp)]
        return selected.reshape(output_size, values.shape[-1])

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(block)
        values = xp.asarray(block)
        values = values.reshape(output_shape + (values.shape[-1],))
        result = xp.zeros(input_shape + (values.shape[-1],), dtype=values.dtype)
        indexer = _indexer(xp)
        if hasattr(result, "at"):
            update = result.at[indexer]
            result = update.set(values) if scalar_selection else update.add(values)
        elif scalar_selection:
            result[indexer] = values
        else:
            np.add.at(result, indexer, values)
        return result.reshape(input_size, -1)

    def normal_matrix_diag(row_scale=None) -> np.ndarray:
        diagonal = np.zeros(input_shape, dtype=dtype)
        values = (
            1.0 if row_scale is None else np.abs(to_numpy(row_scale).reshape(output_shape)) ** 2
        )
        if scalar_selection:
            diagonal[scalar_indexer] = values
        else:
            np.add.at(diagonal, _indexer(np), values)
        return diagonal.reshape(-1)

    def dense_array(xp: Any) -> Any:
        input_indices = xp.arange(input_size).reshape(input_shape)
        selected = input_indices[_indexer(xp)].reshape(-1)
        dense = xp.zeros((output_size, input_size), dtype=dtype)
        rows = xp.arange(output_size)
        if hasattr(dense, "at"):
            return dense.at[rows, selected].set(1)
        dense[rows, selected] = 1
        return dense

    def sparse_matrix():
        selected = np.arange(input_size).reshape(input_shape)[_indexer(np)].reshape(-1)
        return scipy.sparse.csr_matrix(
            (np.ones(output_size, dtype=dtype), (np.arange(output_size), selected)),
            shape=(output_size, input_size),
        )

    return LinearMap(
        shape=(output_size, input_size),
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dense_array=dense_array,
        normal_matrix_diag=normal_matrix_diag,
        input_shape=input_shape,
        output_shape=output_shape,
        sparse_matrix=sparse_matrix,
    )


def is_identity_linear_map(
    value: Any,
    *,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> bool:
    """Return whether ``value`` is an explicit diagonal identity map."""
    try:
        linear_map = as_linear_map(value, input_shape=input_shape, output_shape=output_shape)
    except (TypeError, ValueError):
        return False
    if linear_map.shape[0] != linear_map.shape[1]:
        return False
    if linear_map.input_shape != linear_map.output_shape:
        return False
    if linear_map._is_identity:
        return True
    if not linear_map.is_diagonal:
        return False
    diagonal = np.asarray(linear_map.diagonal(backend="numpy"))
    return bool(np.array_equal(diagonal, np.ones_like(diagonal)))


def _zero_row_linear_map(input_shape: tuple[int, ...]) -> LinearMap:
    """Return the neutral zero-row map for vertical stacking."""
    input_shape = tuple(input_shape)
    input_size = math.prod(input_shape)

    def matvec(vec: Any) -> Any:
        xp = get_array_module(vec)
        return xp.zeros((0,), dtype=xp.asarray(vec).dtype)

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(vec)
        return xp.zeros((input_size,), dtype=xp.asarray(vec).dtype)

    def normal_matrix_diag(row_scale=None) -> np.ndarray:
        return np.zeros(input_size)

    return LinearMap(
        shape=(0, input_size),
        dtype=np.float64,
        matvec=matvec,
        rmatvec=rmatvec,
        normal_matrix_diag=normal_matrix_diag,
        output_shape=(0,),
        input_shape=input_shape,
    )


def vstack_linear_maps(
    maps: Sequence[Any], *, input_shape: tuple[int, ...] | None = None
) -> LinearMap:
    """Return one map formed by vertically stacking row maps."""
    row_maps = tuple(
        as_linear_map(item, input_shape=input_shape)
        if input_shape is not None
        else as_linear_map(item)
        for item in maps
    )
    if len(row_maps) == 1:
        return row_maps[0]
    if not row_maps:
        if input_shape is None:
            raise ValueError("input_shape is required when stacking no maps.")
        return _zero_row_linear_map(input_shape)

    first = row_maps[0]
    input_size = first.shape[1]
    common_input_shape = first.input_shape
    for row_map in row_maps[1:]:
        if row_map.shape[1] != input_size or row_map.input_shape != common_input_shape:
            raise ValueError("Stacked maps must share one input shape.")

    output_size = sum(row_map.shape[0] for row_map in row_maps)
    dtype = np.result_type(*(row_map.dtype for row_map in row_maps))
    backend_operands = tuple(
        operand for row_map in row_maps for operand in row_map.backend_operands
    )

    def array_module_for(value: Any) -> Any:
        return get_array_module(value, *backend_operands)

    def matmat(block: Any) -> Any:
        xp = array_module_for(block)
        block_arr = xp.asarray(block).reshape(input_size, -1)
        return xp.vstack([xp.asarray(row_map.matmat(block_arr)) for row_map in row_maps])

    def rmatmat(block: Any) -> Any:
        xp = array_module_for(block)
        block_arr = xp.asarray(block).reshape(output_size, -1)
        accum = xp.zeros((input_size, block_arr.shape[1]), dtype=block_arr.dtype)
        row = 0
        for row_map in row_maps:
            part = block_arr[row : row + row_map.shape[0], :]
            accum = accum + xp.asarray(row_map.rmatmat(part))
            row += row_map.shape[0]
        return accum

    def matvec(vec: Any) -> Any:
        xp = array_module_for(vec)
        return matmat(xp.asarray(vec).reshape(input_size, 1)).reshape(-1)

    def rmatvec(vec: Any) -> Any:
        xp = array_module_for(vec)
        return rmatmat(xp.asarray(vec).reshape(output_size, 1)).reshape(-1)

    def dense_array(xp: Any) -> Any:
        return xp.vstack([xp.asarray(row_map._dense_array(xp)) for row_map in row_maps])

    def normal_matrix_diag(row_scale=None) -> np.ndarray:
        diag = np.zeros(input_size, dtype=np.result_type(dtype, np.float64))
        row = 0
        for row_map in row_maps:
            scale = None if row_scale is None else row_scale[row : row + row_map.shape[0]]
            diag += row_map.normal_matrix_diag(row_scale=scale)
            row += row_map.shape[0]
        return diag

    def normal_matrix(xp, row_scale):
        normal = None
        row = 0
        for row_map in row_maps:
            scale = None if row_scale is None else row_scale[row : row + row_map.shape[0]]
            if row_map.is_diagonal:
                diagonal = row_map._diagonal_array(xp)
                if scale is not None:
                    diagonal = diagonal * xp.asarray(scale)
                increment = xp.abs(diagonal) ** 2
                if normal is None:
                    normal = xp.diag(increment).astype(dtype)
                else:
                    indices = xp.diag_indices(input_size)
                    if xp is np:
                        normal[indices] += increment
                    else:
                        normal = normal.at[indices].add(increment)
            else:
                matrix = row_map._normal_matrix(xp, scale)
                normal = (
                    xp.array(matrix, dtype=dtype, copy=True) if normal is None else normal + matrix
                )
            row += row_map.shape[0]
        return normal

    return LinearMap(
        shape=(output_size, input_size),
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dense_array=dense_array,
        normal_matrix_diag=normal_matrix_diag,
        normal_matrix=normal_matrix,
        backend_operands=backend_operands,
        output_shape=(output_size,),
        input_shape=common_input_shape,
        sparse_matrix=(
            lambda: scipy.sparse.vstack([row_map.to_sparse_matrix() for row_map in row_maps])
        )
        if all(row_map.is_sparse for row_map in row_maps)
        else None,
    )


def _linear_map_from_linear_operator(
    op: ScipyLinearOperator,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> LinearMap:
    shape = tuple(int(dim) for dim in op.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = op.dtype or np.float64

    def matvec(vec: Any) -> Any:
        return op.matvec(np.asarray(vec).reshape(shape[1]))

    def rmatvec(vec: Any) -> Any:
        return op.rmatvec(np.asarray(vec).reshape(shape[0]))

    def matmat(block: Any) -> Any:
        return op.matmat(np.asarray(block).reshape(shape[1], -1))

    def rmatmat(block: Any) -> Any:
        return op.rmatmat(np.asarray(block).reshape(shape[0], -1))

    return LinearMap(
        shape=shape,
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        output_shape=out_shape,
        input_shape=in_shape,
    )


def _linear_map_from_scipy_sparse(
    op: scipy.sparse.spmatrix,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> LinearMap:
    sparse = op.tocsr()
    adjoint = sparse.conjugate().transpose().tocsr()
    shape = tuple(int(dim) for dim in sparse.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = sparse.dtype
    jax_sparse_operators = None

    def jax_operators(xp):
        """Transfer sparse structure to JAX once, without densifying it."""
        nonlocal jax_sparse_operators
        if jax_sparse_operators is None:
            from jax import ensure_compile_time_eval
            from jax.experimental.sparse import BCOO

            def as_bcoo(matrix):
                coo = matrix.tocoo()
                indices = np.column_stack([coo.row, coo.col])
                return BCOO(
                    (xp.asarray(coo.data), xp.asarray(indices)),
                    shape=coo.shape,
                    indices_sorted=True,
                    unique_indices=True,
                )

            # SciPy values are static, even when first used under JIT.
            # Cache concrete device arrays, never temporary tracers.
            with ensure_compile_time_eval():
                jax_sparse_operators = as_bcoo(sparse), as_bcoo(adjoint)
        return jax_sparse_operators

    def apply(matrix, values, input_size, *, block=False, use_adjoint=False):
        xp = _runtime_array_module(values)
        values = xp.asarray(values)
        values = (
            values.reshape(input_size, -1)
            if block and values.ndim > 1
            else values.reshape(input_size)
        )
        if xp is np:
            return matrix @ values
        forward, reverse = jax_operators(xp)
        return (reverse if use_adjoint else forward) @ values

    def matvec(vec: Any) -> Any:
        return apply(sparse, vec, shape[1])

    def rmatvec(vec: Any) -> Any:
        return apply(adjoint, vec, shape[0], use_adjoint=True)

    def matmat(block: Any) -> Any:
        return apply(sparse, block, shape[1], block=True)

    def rmatmat(block: Any) -> Any:
        return apply(adjoint, block, shape[0], block=True, use_adjoint=True)

    def dense_array(xp: Any) -> Any:
        return xp.asarray(sparse.toarray())

    def normal_matrix_diag(row_scale=None) -> np.ndarray:
        squared = sparse.multiply(sparse.conjugate())
        if row_scale is not None:
            squared = squared.multiply(np.abs(to_numpy(row_scale))[:, None] ** 2)
        return np.asarray(squared.sum(axis=0)).reshape(-1).real

    return LinearMap(
        shape=shape,
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dense_array=dense_array,
        normal_matrix_diag=normal_matrix_diag,
        output_shape=out_shape,
        input_shape=in_shape,
        sparse_matrix=lambda: sparse,
    )


def _linear_map_from_jax_sparse(
    op: Any,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> LinearMap:
    shape = tuple(int(dim) for dim in op.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = op.dtype
    transposed = op.T
    adjoint = type(transposed)(
        (transposed.data.conj(), transposed.indices), shape=transposed.shape
    )
    backend_operands = tuple(
        operand
        for operand in (getattr(op, "data", None), getattr(op, "indices", None))
        if operand is not None
    )

    def matvec(vec: Any) -> Any:
        xp = get_array_module(vec, *backend_operands)
        return op @ xp.asarray(vec).reshape(shape[1])

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(vec, *backend_operands)
        return adjoint @ xp.asarray(vec).reshape(shape[0])

    def matmat(block: Any) -> Any:
        xp = get_array_module(block, *backend_operands)
        return op @ xp.asarray(block).reshape(shape[1], -1)

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(block, *backend_operands)
        return adjoint @ xp.asarray(block).reshape(shape[0], -1)

    def dense_array(xp: Any) -> Any:
        return xp.asarray(op.todense())

    scalar_entries = op.data.ndim == 1 and op.indices.ndim == 2 and op.indices.shape[1] == 2

    def sparse_matrix():
        data, indices = to_numpy(op.data), to_numpy(op.indices)
        # BCOO pads unused storage with out-of-bounds indices. Coalesce
        # duplicates before squaring entries for the normal diagonal.
        valid = np.all((indices >= 0) & (indices < shape), axis=1)
        return scipy.sparse.coo_matrix(
            (data[valid], (indices[valid, 0], indices[valid, 1])), shape=shape
        ).tocsr()

    def normal_matrix_diag(row_scale=None) -> np.ndarray:
        if not scalar_entries:
            return _normal_matrix_diag_from_matmat(shape, dtype, matmat, row_scale)
        sparse = sparse_matrix()
        squared = sparse.multiply(sparse.conjugate())
        if row_scale is not None:
            squared = squared.multiply(np.abs(to_numpy(row_scale))[:, None] ** 2)
        return np.asarray(squared.sum(axis=0)).reshape(-1).real

    return LinearMap(
        shape=shape,
        dtype=dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dense_array=dense_array,
        normal_matrix_diag=normal_matrix_diag,
        backend_operands=backend_operands,
        output_shape=out_shape,
        input_shape=in_shape,
        sparse_matrix=sparse_matrix if scalar_entries else None,
    )


def _linear_map_from_array(
    arr: Any,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> LinearMap:
    """Convert a dense array-shaped value into a ``LinearMap``."""
    if arr.ndim == 1:
        size = int(arr.size)
        if input_shape is not None and math.prod(input_shape) != size:
            raise ValueError(f"1-D operator size {size} mismatch with input {input_shape}.")
        if output_shape is not None and math.prod(output_shape) != size:
            raise ValueError(f"1-D operator size {size} mismatch with output {output_shape}.")
        return diagonal_linear_map(arr, input_shape=input_shape, output_shape=output_shape)

    if arr.ndim < 2:
        raise ValueError("Operators must be at least 1-D.")

    if arr.ndim == 2 and input_shape is None and output_shape is None:
        return _linear_map_from_dense(arr)

    inferred_input = (arr.shape[-1],) if input_shape is None else input_shape
    flat_in = math.prod(inferred_input)
    total_elements = int(arr.size)
    if output_shape is None:
        flat_out = total_elements // flat_in
        if flat_out * flat_in != total_elements:
            raise ValueError(
                f"Operator with shape {arr.shape} incompatible with inferred input "
                f"{inferred_input}."
            )
        input_ndim = len(inferred_input)
        if input_ndim <= arr.ndim and tuple(arr.shape[-input_ndim:]) == inferred_input:
            inferred_output = tuple(arr.shape[:-input_ndim])
        else:
            inferred_output = (flat_out,)
    else:
        flat_out = math.prod(output_shape)
        if flat_out * flat_in != total_elements:
            raise ValueError(
                f"Operator with shape {arr.shape} incompatible with provided shapes "
                f"{output_shape} -> {inferred_input}."
            )
        inferred_output = output_shape
    return _linear_map_from_dense(
        arr.reshape(flat_out, flat_in), input_shape=inferred_input, output_shape=inferred_output
    )


def as_linear_map(
    op: Any,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> LinearMap:
    """Convert supported operator types into a ``LinearMap``."""
    if isinstance(op, LinearMap):
        if input_shape is None and output_shape is None:
            return op
        out_shape, in_shape = _map_shapes(
            op.shape,
            input_shape if input_shape is not None else op.input_shape,
            output_shape if output_shape is not None else op.output_shape,
        )
        if out_shape == op.output_shape and in_shape == op.input_shape:
            return op
        relabeled = LinearMap(
            shape=op.shape,
            dtype=op.dtype,
            matvec=op._matvec,
            rmatvec=op._rmatvec,
            matmat=op._matmat,
            rmatmat=op._rmatmat,
            dense_array=op._dense_array_func,
            diagonal=op._diagonal_array_func,
            normal_matrix_diag=op._normal_matrix_diag,
            normal_matrix=op._normal_matrix_func,
            sparse_matrix=op._sparse_matrix_func,
            backend_operands=op._backend_operands,
            output_shape=out_shape,
            input_shape=in_shape,
        )
        object.__setattr__(relabeled, "_is_identity", op._is_identity)
        object.__setattr__(relabeled, "_einsum_map", op._einsum_map)
        object.__setattr__(relabeled, "_dense_tensor", op._dense_tensor)
        # The flat matrix is unchanged by shaped metadata. Share any dense
        # materialization already paid for.
        object.__setattr__(relabeled, "_dense_cache", op._dense_cache)
        object.__setattr__(relabeled, "_sparse_cache", op._sparse_cache)
        return relabeled

    op_type = str(type(op))
    is_jax_sparse = "jax.experimental.sparse" in op_type or (
        "jax" in op_type and hasattr(op, "todense") and hasattr(op, "indices")
    )
    if is_jax_sparse:
        return _linear_map_from_jax_sparse(op, input_shape=input_shape, output_shape=output_shape)

    if isinstance(op, ScipyLinearOperator):
        return _linear_map_from_linear_operator(
            op, input_shape=input_shape, output_shape=output_shape
        )

    if scipy.sparse.issparse(op):
        return _linear_map_from_scipy_sparse(
            op, input_shape=input_shape, output_shape=output_shape
        )

    arr = _dense_array_candidate(op)
    return _linear_map_from_array(arr, input_shape=input_shape, output_shape=output_shape)


def _weighted_cross_product(left, right, weights):
    """Return ``left* diag(weights) right`` in bounded row blocks."""
    xp = get_array_module(left, right, weights)
    left = xp.asarray(left)
    right = xp.asarray(right)
    weights = xp.asarray(weights)
    if left.shape[0] != right.shape[0] or weights.shape != (left.shape[0],):
        raise ValueError("Weighted cross-product operands have incompatible shapes.")
    dtype = xp.result_type(left.dtype, right.dtype, weights.dtype)
    result = xp.zeros((left.shape[1], right.shape[1]), dtype=dtype)
    bytes_per_row = max(1, right.shape[1] * np.dtype(dtype).itemsize)
    rows_per_block = max(1, _WEIGHTED_PRODUCT_WORK_BYTES // bytes_per_row)
    for start in range(0, left.shape[0], rows_per_block):
        stop = min(left.shape[0], start + rows_per_block)
        weighted_right = weights[start:stop, None] * right[start:stop]
        result += left[start:stop].T.conj() @ weighted_right
    return result
