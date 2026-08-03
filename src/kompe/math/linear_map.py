"""Backend-aware linear-operator wrapper."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from kompe.math.backend import (
    JAX_AVAILABLE,
    asarray,
    block_until_ready,
    get_array_module,
    to_numpy,
    use_jax,
)

MatrixShape: TypeAlias = tuple[int, int]
VectorizedMapFunc: TypeAlias = Callable[[Any], Any]
MatrixBackend: TypeAlias = Literal["numpy", "jax"]


def _array_module_for_matrix_backend(backend: MatrixBackend | None = None) -> Any:
    """Return the array module for explicit matrix materialization."""
    if backend is None:
        return None
    if not isinstance(backend, str):
        raise TypeError("backend must be None, 'numpy', or 'jax'.")

    normalized = backend.strip().lower()
    if normalized == "numpy":
        return np
    if normalized == "jax":
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not installed; cannot materialize on JAX.")
        import jax.numpy as jnp

        return jnp
    raise ValueError(f"Unknown matrix backend {backend!r}. Use None, 'numpy', or 'jax'.")


@dataclass(frozen=True)
class LinearMap:
    """Backend-agnostic linear map with flattened matrix operations."""

    shape: MatrixShape
    dtype: Any
    _matvec: VectorizedMapFunc = field(repr=False)
    _rmatvec: VectorizedMapFunc = field(repr=False)
    _matmat: VectorizedMapFunc | None = field(default=None, repr=False)
    _rmatmat: VectorizedMapFunc | None = field(default=None, repr=False)
    _dense_array_func: Callable[[Any], Any] | None = field(default=None, repr=False)
    _diagonal_array_func: Callable[[Any], Any] | None = field(default=None, repr=False)
    _normal_matrix_diag: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    _backend_context: tuple[Any, ...] = field(default=(), repr=False)
    _is_noop: bool = field(default=False, repr=False)
    _einsum_map: Any = field(default=None, repr=False, compare=False)
    _dense_tensor: Any = field(default=None, repr=False, compare=False)
    output_shape: tuple[int, ...] | None = None
    input_shape: tuple[int, ...] | None = None
    _dense_cache: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _array_cache: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate shaped metadata and fill flat defaults."""
        output_shape, input_shape = _map_shapes(self.shape, self.input_shape, self.output_shape)
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "input_shape", input_shape)

    @property
    def ndim(self) -> int:
        """Dimensionality of the linear map."""
        return 2

    @property
    def backend_context(self) -> tuple[Any, ...]:
        """Closed-over operands used for backend selection."""
        return self._backend_context

    def array_module(self, *operands: Any) -> Any:
        """Return the array module implied by operands and this map."""
        return get_array_module(*operands, *self._backend_context)

    @staticmethod
    def _dense_cache_key(xp: Any) -> str:
        """Return a stable cache key for one array module."""
        return getattr(xp, "__name__", repr(xp))

    def _cached_dense(self, xp: Any) -> Any | None:
        """Return cached dense materialization for ``xp`` if present."""
        return self._dense_cache.get(self._dense_cache_key(xp))

    def _store_dense(self, xp: Any, dense: Any) -> None:
        """Store dense materialization for ``xp``."""
        self._dense_cache[self._dense_cache_key(xp)] = dense

    def _cached_array(self, xp: Any) -> Any | None:
        """Return cached shaped materialization for ``xp``."""
        return self._array_cache.get(self._dense_cache_key(xp))

    def _store_array(self, xp: Any, array: Any) -> None:
        """Store shaped materialization for ``xp``."""
        self._array_cache[self._dense_cache_key(xp)] = array

    def matvec(self, x: Any) -> Any:
        """Apply this map to one flattened vector."""
        xp = self.array_module(x)
        dense = self._cached_dense(xp)
        if dense is not None:
            x_arr = xp.asarray(x).reshape(self.shape[1])
            return dense @ x_arr
        return self._matvec(x)

    def rmatvec(self, y: Any) -> Any:
        """Apply the adjoint map to one flattened vector."""
        xp = self.array_module(y)
        dense = self._cached_dense(xp)
        if dense is not None:
            y_arr = xp.asarray(y).reshape(self.shape[0])
            return xp.swapaxes(xp.conjugate(dense), -2, -1) @ y_arr
        return self._rmatvec(y)

    def matmat(self, x_block: Any) -> Any:
        """Apply this map to a block of column vectors."""
        xp = self.array_module(x_block)
        dense = self._cached_dense(xp)
        if dense is not None:
            x_arr = xp.asarray(x_block)
            if x_arr.ndim == 1:
                return dense @ x_arr.reshape(self.shape[1])
            return dense @ x_arr.reshape(self.shape[1], -1)
        if self._matmat is not None:
            return self._matmat(x_block)
        x_arr = xp.asarray(x_block)
        if x_arr.ndim == 1:
            return self.matvec(x_arr)
        outputs = [self.matvec(x_arr[:, i]) for i in range(x_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def rmatmat(self, y_block: Any) -> Any:
        """Apply the adjoint map to a block of column vectors."""
        xp = self.array_module(y_block)
        dense = self._cached_dense(xp)
        if dense is not None:
            y_arr = xp.asarray(y_block)
            adjoint = xp.swapaxes(xp.conjugate(dense), -2, -1)
            if y_arr.ndim == 1:
                return adjoint @ y_arr.reshape(self.shape[0])
            return adjoint @ y_arr.reshape(self.shape[0], -1)
        if self._rmatmat is not None:
            return self._rmatmat(y_block)
        y_arr = xp.asarray(y_block)
        if y_arr.ndim == 1:
            return self.rmatvec(y_arr)
        outputs = [self.rmatvec(y_arr[:, i]) for i in range(y_arr.shape[1])]
        return xp.stack(outputs, axis=1)

    def _dense_array(self, xp: Any = None) -> Any:
        """Materialize this map as a dense 2-D matrix on ``xp``."""
        xp = self.array_module() if xp is None else xp
        cached = self._cached_dense(xp)
        if cached is not None:
            return cached

        cached_array = self._cached_array(xp)
        if cached_array is not None:
            dense = xp.reshape(cached_array, self.shape)
            self._store_dense(xp, dense)
            return dense

        if self._dense_array_func is not None:
            dense = self._dense_array_func(xp)
            self._store_dense(xp, dense)
            return dense

        eye_dtype = np.result_type(self.dtype, np.float64)
        if self.shape[0] < self.shape[1]:
            output_identity = xp.eye(self.shape[0], dtype=eye_dtype)
            adjoint = self.rmatmat(output_identity)
            dense = xp.swapaxes(xp.conjugate(adjoint), -2, -1)
        else:
            input_identity = xp.eye(self.shape[1], dtype=eye_dtype)
            dense = self.matmat(input_identity)
        dense = np.asarray(dense) if xp is np else xp.asarray(dense)
        self._store_dense(xp, dense)
        return dense

    def _explicit_array(self, xp: Any = None) -> Any:
        """Materialize this map as a shaped dense array on ``xp``."""
        xp = self.array_module() if xp is None else xp
        cached = self._cached_array(xp)
        if cached is not None:
            return cached
        dense = self._dense_array(xp)
        array = xp.reshape(dense, self.output_shape + self.input_shape)
        self._store_array(xp, array)
        return array

    def to_matrix(self, *, backend: MatrixBackend | None = None) -> Any:
        """Materialize this map as an explicit flat 2-D matrix."""
        xp = _array_module_for_matrix_backend(backend)
        self._explicit_array(xp)
        return block_until_ready(self._dense_array(xp))

    @property
    def array(self) -> Any:
        """Lazily materialized shaped operator array."""
        return block_until_ready(self._explicit_array())

    def diagonal(self, *, backend: MatrixBackend | None = None) -> Any:
        """Return diagonal scale values for a diagonal map."""
        xp = _array_module_for_matrix_backend(backend)
        return block_until_ready(self._diagonal_array(xp))

    def _diagonal_array(self, xp: Any = None) -> Any:
        """Return exact diagonal scale values on ``xp``."""
        xp = self.array_module() if xp is None else xp
        if self._diagonal_array_func is not None:
            return self._diagonal_array_func(xp)
        if self.shape[0] != self.shape[1]:
            raise ValueError("Diagonal values require a square operator.")

        dense = np.asarray(self.to_matrix(backend="numpy"))
        diagonal = np.diag(dense)
        if not np.allclose(dense, np.diag(diagonal), rtol=0.0, atol=0.0):
            raise ValueError("Operator is not diagonal.")
        return xp.asarray(diagonal)

    def normal_matrix_diag(self) -> np.ndarray:
        """Compute ``diag(A* A)`` for this map."""
        if self._normal_matrix_diag is not None:
            return np.asarray(self._normal_matrix_diag())
        if self._diagonal_array_func is not None:
            return np.abs(np.asarray(self.diagonal(backend="numpy"))) ** 2
        if self._cached_dense(np) is not None or self._dense_array_func is not None:
            dense = np.asarray(self.to_matrix(backend="numpy"))
            return np.sum(np.abs(dense) ** 2, axis=0)
        return _normal_matrix_diag_from_matmat(self.shape, self.dtype, self.matmat)

    def __matmul__(self, other: Any) -> Any:
        """Apply to arrays or compose with another operator."""
        if not scipy.sparse.issparse(other) and not _looks_like_operator(other):
            arr = asarray(other)
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
        if self._is_noop:
            return as_linear_map(
                other_map, input_shape=other_map.input_shape, output_shape=self.output_shape
            )
        if other_map._is_noop:
            return as_linear_map(
                self, input_shape=other_map.input_shape, output_shape=self.output_shape
            )
        composed_einsum = self._compose_einsum_matmul(other_map)
        if composed_einsum is not None:
            return composed_einsum.to_linear_map()
        return self._composed_linear_map(other_map)

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

        def dense_array(xp: Any) -> Any:
            return self._composition_dense_array(other_map, xp)

        dtype = np.promote_types(self.dtype, other_map.dtype)

        diagonal_array = None
        if self_is_diagonal and other_is_diagonal:

            def diagonal_array(xp: Any) -> Any:
                return self._diagonal_array(xp) * other_map._diagonal_array(xp)

        def normal_matrix_diag() -> np.ndarray:
            return self._composition_normal_matrix_diag(other_map, dtype, matmat)

        return LinearMap(
            shape=(self.shape[0], other_map.shape[1]),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _dense_array_func=dense_array,
            _diagonal_array_func=diagonal_array,
            _normal_matrix_diag=normal_matrix_diag,
            _backend_context=self._backend_context + other_map._backend_context,
            output_shape=self.output_shape,
            input_shape=other_map.input_shape,
        )

    def _composition_dense_array(self, other_map: LinearMap, xp: Any) -> Any:
        """Materialize a composition, preserving diagonal structure."""
        self_is_diagonal = self._diagonal_array_func is not None
        other_is_diagonal = other_map._diagonal_array_func is not None
        if self_is_diagonal and other_is_diagonal:
            diagonal = self._diagonal_array(xp) * other_map._diagonal_array(xp)
            return xp.diag(diagonal)
        if self_is_diagonal:
            return self._diagonal_array(xp).reshape(-1, 1) * other_map._dense_array(xp)
        if other_is_diagonal:
            return self._dense_array(xp) * other_map._diagonal_array(xp).reshape(1, -1)
        eye_dtype = np.result_type(self.dtype, other_map.dtype, np.float64)
        if self.shape[0] < other_map.shape[1]:
            output_identity = xp.eye(self.shape[0], dtype=eye_dtype)
            adjoint = other_map.rmatmat(self.rmatmat(output_identity))
            return xp.swapaxes(xp.conjugate(adjoint), -2, -1)
        input_identity = xp.eye(other_map.shape[1], dtype=eye_dtype)
        return xp.asarray(self.matmat(other_map.matmat(input_identity)))

    def _composition_normal_matrix_diag(self, other_map, dtype, matmat):
        """Return the normal diagonal of a lazy composition."""
        if other_map._diagonal_array_func is not None:
            diagonal = np.asarray(other_map.diagonal(backend="numpy"))
            return np.abs(diagonal) ** 2 * self.normal_matrix_diag()
        return _normal_matrix_diag_from_matmat((self.shape[0], other_map.shape[1]), dtype, matmat)

    def _compose_einsum_matmul(self, other_map: LinearMap) -> Any:
        """Return a symbolic einsum composition, when safe."""
        self_is_diagonal = self._diagonal_array_func is not None
        other_is_diagonal = other_map._diagonal_array_func is not None
        if self_is_diagonal and other_is_diagonal:
            return None
        if self_is_diagonal:
            right_einsum = other_map._composition_einsum_map()
            if (
                right_einsum is None
                or self.output_shape != self.input_shape
                or self.output_shape != other_map.output_shape
                or self.output_shape != right_einsum.output_shape
            ):
                return None
            try:
                from kompe.math.einsum import compose_diagonal_einsum_map

                return compose_diagonal_einsum_map(
                    self._diagonal_array(), self.output_shape, right_einsum, side="left"
                )
            except ValueError:
                return None
        if other_is_diagonal:
            left_einsum = self._composition_einsum_map()
            if (
                left_einsum is None
                or other_map.output_shape != other_map.input_shape
                or other_map.input_shape != self.input_shape
                or other_map.input_shape != left_einsum.input_shape
            ):
                return None
            try:
                from kompe.math.einsum import compose_diagonal_einsum_map

                return compose_diagonal_einsum_map(
                    other_map._diagonal_array(), other_map.input_shape, left_einsum, side="right"
                )
            except ValueError:
                return None

        left_einsum = self._composition_einsum_map()
        right_einsum = other_map._composition_einsum_map()
        if left_einsum is None or right_einsum is None:
            return None
        if (
            self.output_shape != left_einsum.output_shape
            or self.input_shape != left_einsum.input_shape
            or other_map.output_shape != right_einsum.output_shape
            or other_map.input_shape != right_einsum.input_shape
        ):
            return None
        try:
            from kompe.math.einsum import compose_einsum_maps

            return compose_einsum_maps(left_einsum, right_einsum)
        except ValueError:
            return None

    def _composition_einsum_map(self) -> Any:
        """Return an einsum view for composition, when safe."""
        if (
            self._einsum_map is not None
            and self._einsum_map.output_shape == self.output_shape
            and self._einsum_map.input_shape == self.input_shape
        ):
            return self._einsum_map
        if self._dense_tensor is None:
            return None
        if tuple(getattr(self._dense_tensor, "shape", ())) != (
            self.output_shape + self.input_shape
        ):
            return None
        from kompe.math.einsum import dense_tensor_einsum_map

        return dense_tensor_einsum_map(
            self._dense_tensor, output_shape=self.output_shape, input_shape=self.input_shape
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

        def normal_matrix_diag() -> np.ndarray:
            return _normal_matrix_diag_from_matmat(self.shape, dtype, matmat)

        return LinearMap(
            shape=self.shape,
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _dense_array_func=dense_array,
            _normal_matrix_diag=normal_matrix_diag,
            _backend_context=self._backend_context + other_map._backend_context,
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
        if (
            self._einsum_map is not None
            and self._einsum_map.output_shape == self.output_shape
            and self._einsum_map.input_shape == self.input_shape
        ):
            from kompe.math.einsum import scale_einsum_map

            return scale_einsum_map(self._einsum_map, scalar).to_linear_map()

        scaled_dense_tensor = None
        if (
            self._dense_tensor is not None
            and tuple(getattr(self._dense_tensor, "shape", ()))
            == self.output_shape + self.input_shape
        ):
            scaled_dense_tensor = self._dense_tensor * scalar

        def matvec(x: Any) -> Any:
            return self.matvec(x) * scalar

        def rmatvec(y: Any) -> Any:
            return self.rmatvec(y) * np.conj(scalar)

        def matmat(x: Any) -> Any:
            return self.matmat(x) * scalar

        def rmatmat(y: Any) -> Any:
            return self.rmatmat(y) * np.conj(scalar)

        def dense_array(xp: Any) -> Any:
            return self._dense_array(xp) * scalar

        def normal_matrix_diag() -> np.ndarray:
            return np.abs(scalar) ** 2 * self.normal_matrix_diag()

        def diagonal_array(xp: Any) -> Any:
            return self._diagonal_array(xp) * scalar

        return LinearMap(
            shape=self.shape,
            dtype=np.result_type(self.dtype, scalar),
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _dense_array_func=dense_array,
            _diagonal_array_func=(
                diagonal_array if self._diagonal_array_func is not None else None
            ),
            _normal_matrix_diag=normal_matrix_diag,
            _backend_context=self._backend_context,
            _dense_tensor=scaled_dense_tensor,
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


def _looks_like_operator(value: Any) -> bool:
    return isinstance(value, (LinearMap, ScipyLinearOperator)) or hasattr(value, "matvec")


def _runtime_array_module(*values: Any) -> Any:
    """Select JAX only when an operand is already a JAX array."""
    if any("jax" in type(value).__module__ for value in values):
        return get_array_module(*values)
    return np


def _normal_matrix_diag_from_matmat(
    shape: MatrixShape, dtype: Any, matmat: Callable[[Any], Any]
) -> np.ndarray:
    """Compute ``diag(A* A)`` from bounded identity blocks."""
    n_cols = shape[1]
    work_dtype = np.result_type(dtype, np.float64)
    diag = np.zeros(n_cols, dtype=work_dtype)
    block_size = min(32, max(1, n_cols))
    block = np.zeros((n_cols, block_size), dtype=work_dtype)
    for start in range(0, n_cols, block_size):
        stop = min(n_cols, start + block_size)
        cols = stop - start
        block[:, :cols] = 0
        block[start:stop, :cols] = np.eye(cols, dtype=work_dtype)
        res = np.asarray(matmat(block[:, :cols]))
        diag[start:stop] = np.sum(np.abs(res) ** 2, axis=0).real
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
        return xp.matmul(xp.swapaxes(xp.conjugate(mat_arr), -2, -1), vec_arr)

    def matmat(block: Any) -> Any:
        xp = _runtime_array_module(mat_array, block)
        mat_arr = xp.asarray(mat_array)
        block_arr = xp.asarray(block).reshape(shape[1], -1)
        return xp.matmul(mat_arr, block_arr)

    def rmatmat(block: Any) -> Any:
        xp = _runtime_array_module(mat_array, block)
        mat_arr = xp.asarray(mat_array)
        block_arr = xp.asarray(block).reshape(shape[0], -1)
        adjoint = xp.swapaxes(xp.conjugate(mat_arr), -2, -1)
        return xp.matmul(adjoint, block_arr)

    def normal_matrix_diag() -> np.ndarray:
        mat_np = to_numpy(mat_array)
        return np.sum(np.abs(mat_np) ** 2, axis=0)

    def dense_array(xp: Any) -> Any:
        return xp.asarray(mat_array)

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        _backend_context=(mat_array,),
        _dense_tensor=mat_array.reshape(out_shape + in_shape),
        output_shape=out_shape,
        input_shape=in_shape,
    )


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

    def normal_matrix_diag() -> np.ndarray:
        return np.abs(to_numpy(diag_array)) ** 2

    def dense_array(xp: Any) -> Any:
        return xp.diag(xp.asarray(diag_array))

    def diagonal_array(xp: Any) -> Any:
        return xp.asarray(diag_array).reshape(size)

    return LinearMap(
        shape=(size, size),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _diagonal_array_func=diagonal_array,
        _normal_matrix_diag=normal_matrix_diag,
        _backend_context=(diag_array,),
        output_shape=out_shape,
        input_shape=in_shape,
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

    def normal_matrix_diag() -> np.ndarray:
        return np.ones(size, dtype=dtype)

    return LinearMap(
        shape=(size, size),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=matvec,
        _matmat=matmat,
        _rmatmat=matmat,
        _dense_array_func=dense_array,
        _diagonal_array_func=diagonal_array,
        _normal_matrix_diag=normal_matrix_diag,
        _is_noop=True,
        output_shape=value_shape,
        input_shape=value_shape,
    )


def pointwise_matrix_linear_map(matrix: Any) -> LinearMap:
    """Return a pointwise component map.

    ``matrix`` has shape
    ``(n_output_components, n_input_components, *points)`` and maps
    arrays shaped ``(n_input_components, *points)`` to
    ``(n_output_components, *points)``.
    """
    matrix_array = _dense_array_candidate(matrix)
    if matrix_array.ndim < 2:
        raise ValueError("pointwise matrix must have at least two component axes.")

    output_components = int(matrix_array.shape[0])
    input_components = int(matrix_array.shape[1])
    point_shape = tuple(int(dim) for dim in matrix_array.shape[2:])
    input_shape = (input_components,) + point_shape
    output_shape = (output_components,) + point_shape
    input_size = int(math.prod(input_shape))
    output_size = int(math.prod(output_shape))
    dtype = matrix_array.dtype

    def matvec(vec: Any) -> Any:
        xp = _runtime_array_module(matrix_array, vec)
        values = xp.asarray(vec).reshape(input_shape)
        result = xp.einsum("ab...,b...->a...", xp.asarray(matrix_array), values, optimize=True)
        return result.reshape(-1)

    def rmatvec(vec: Any) -> Any:
        xp = _runtime_array_module(matrix_array, vec)
        values = xp.asarray(vec).reshape(output_shape)
        result = xp.einsum(
            "ab...,a...->b...", xp.conjugate(xp.asarray(matrix_array)), values, optimize=True
        )
        return result.reshape(-1)

    def matmat(block: Any) -> Any:
        xp = _runtime_array_module(matrix_array, block)
        values = xp.asarray(block).reshape(input_shape + (-1,))
        result = xp.einsum("ab...,b...h->a...h", xp.asarray(matrix_array), values, optimize=True)
        return result.reshape(output_size, -1)

    def rmatmat(block: Any) -> Any:
        xp = _runtime_array_module(matrix_array, block)
        values = xp.asarray(block).reshape(output_shape + (-1,))
        result = xp.einsum(
            "ab...,a...h->b...h", xp.conjugate(xp.asarray(matrix_array)), values, optimize=True
        )
        return result.reshape(input_size, -1)

    def normal_matrix_diag() -> np.ndarray:
        return np.sum(np.abs(to_numpy(matrix_array)) ** 2, axis=0).reshape(-1)

    def dense_array(xp: Any) -> Any:
        point_size = int(math.prod(point_shape))
        matrix_values = xp.asarray(matrix_array).reshape(
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
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        _backend_context=(matrix_array,),
        input_shape=input_shape,
        output_shape=output_shape,
    )


def _normalize_take_selection(input_shape, indices, axis):
    """Return validated shapes, axis, and integer selection indices."""
    input_shape = tuple(int(dim) for dim in input_shape)
    if not input_shape:
        raise ValueError("input_shape must have at least one axis.")
    axis = int(axis)
    if axis < 0:
        axis += len(input_shape)
    if axis < 0 or axis >= len(input_shape):
        raise ValueError(f"axis {axis} is outside input_shape {input_shape}.")

    index_array = np.asarray(indices)
    if index_array.dtype == np.bool_:
        if index_array.shape != (input_shape[axis],):
            raise ValueError("Boolean indices must match the selected input axis length.")
        index_array = np.flatnonzero(index_array)
    index_array = np.asarray(index_array, dtype=int).reshape(-1)
    if np.any(index_array < 0) or np.any(index_array >= input_shape[axis]):
        raise IndexError("take_linear_map indices are outside input_shape.")

    output_shape = list(input_shape)
    output_shape[axis] = int(index_array.size)
    output_shape = tuple(output_shape)
    return input_shape, output_shape, index_array, axis


def take_linear_map(
    input_shape: tuple[int, ...], indices: Any, *, axis: int = -1, dtype: Any = np.float64
) -> LinearMap:
    """Return a map selecting ``indices`` along one shaped axis."""
    input_shape, output_shape, index_array, axis = _normalize_take_selection(
        input_shape, indices, axis
    )
    input_size = int(math.prod(input_shape))
    output_size = int(math.prod(output_shape))
    dtype = np.dtype(dtype)

    def _indexer(xp: Any, *, batched: bool = False):
        idx = xp.asarray(index_array)
        indexer = [slice(None)] * (len(input_shape) + int(batched))
        indexer[axis] = idx
        return tuple(indexer)

    def matvec(vec: Any) -> Any:
        xp = get_array_module(vec)
        values = xp.asarray(vec).reshape(input_shape)
        return xp.take(values, xp.asarray(index_array), axis=axis).reshape(-1)

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(vec)
        values = xp.asarray(vec).reshape(output_shape)
        result = xp.zeros(input_shape, dtype=values.dtype)
        indexer = _indexer(xp)
        if hasattr(result, "at"):
            return result.at[indexer].add(values).reshape(-1)
        np.add.at(result, indexer, values)
        return result.reshape(-1)

    def matmat(block: Any) -> Any:
        xp = get_array_module(block)
        values = xp.asarray(block).reshape(input_shape + (-1,))
        selected = xp.take(values, xp.asarray(index_array), axis=axis)
        return selected.reshape(output_size, -1)

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(block)
        values = xp.asarray(block).reshape(output_shape + (-1,))
        result = xp.zeros(input_shape + (values.shape[-1],), dtype=values.dtype)
        indexer = _indexer(xp, batched=True)
        if hasattr(result, "at"):
            return result.at[indexer].add(values).reshape(input_size, -1)
        np.add.at(result, indexer, values)
        return result.reshape(input_size, -1)

    def normal_matrix_diag() -> np.ndarray:
        diagonal = np.zeros(input_shape, dtype=dtype)
        indexer = [slice(None)] * len(input_shape)
        indexer[axis] = index_array
        np.add.at(diagonal, tuple(indexer), 1.0)
        return diagonal.reshape(-1)

    def dense_array(xp: Any) -> Any:
        input_indices = xp.arange(input_size).reshape(input_shape)
        selected = xp.take(input_indices, xp.asarray(index_array), axis=axis).reshape(-1)
        dense = xp.zeros((output_size, input_size), dtype=dtype)
        rows = xp.arange(output_size)
        if hasattr(dense, "at"):
            return dense.at[rows, selected].set(1)
        dense[rows, selected] = 1
        return dense

    return LinearMap(
        shape=(output_size, input_size),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def is_noop_linear_map(
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
    if getattr(linear_map, "_is_noop", False):
        return True
    if getattr(linear_map, "_diagonal_array_func", None) is None:
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

    def matmat(block: Any) -> Any:
        xp = get_array_module(block)
        block_arr = xp.asarray(block).reshape(input_size, -1)
        return xp.zeros((0, block_arr.shape[1]), dtype=block_arr.dtype)

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(block)
        block_arr = xp.asarray(block)
        num_rhs = 1 if block_arr.ndim == 1 else block_arr.shape[1]
        return xp.zeros((input_size, num_rhs), dtype=block_arr.dtype)

    def dense_array(xp: Any) -> Any:
        return xp.zeros((0, input_size))

    def normal_matrix_diag() -> np.ndarray:
        return np.zeros(input_size)

    return LinearMap(
        shape=(0, input_size),
        dtype=np.float64,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
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
    backend_context = tuple(operand for row_map in row_maps for operand in row_map.backend_context)

    def array_module_for(value: Any) -> Any:
        return get_array_module(value, *backend_context)

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

    def normal_matrix_diag() -> np.ndarray:
        diag = np.zeros(input_size, dtype=np.result_type(dtype, np.float64))
        for row_map in row_maps:
            diag += row_map.normal_matrix_diag()
        return diag

    return LinearMap(
        shape=(output_size, input_size),
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        _backend_context=backend_context,
        output_shape=(output_size,),
        input_shape=common_input_shape,
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
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
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
    dense_cache: dict[str, Any] = {}
    adjoint_dense_cache: dict[str, Any] = {}

    def cached_dense(matrix: scipy.sparse.spmatrix, xp: Any, cache: dict[str, Any]) -> Any:
        key = getattr(xp, "__name__", repr(xp))
        if key not in cache:
            cache[key] = xp.asarray(matrix.toarray())
        return cache[key]

    def apply(matrix, cache, values, input_size):
        xp = _runtime_array_module(values)
        values = xp.asarray(values)
        values = values.reshape(input_size) if values.ndim == 1 else values.reshape(input_size, -1)
        return matrix @ values if xp is np else cached_dense(matrix, xp, cache) @ values

    def matvec(vec: Any) -> np.ndarray:
        return apply(sparse, dense_cache, vec, shape[1])

    def rmatvec(vec: Any) -> np.ndarray:
        return apply(adjoint, adjoint_dense_cache, vec, shape[0])

    def matmat(block: Any) -> np.ndarray:
        return apply(sparse, dense_cache, block, shape[1])

    def rmatmat(block: Any) -> np.ndarray:
        return apply(adjoint, adjoint_dense_cache, block, shape[0])

    def dense_array(xp: Any) -> Any:
        return xp.asarray(sparse.toarray())

    def normal_matrix_diag() -> np.ndarray:
        return np.asarray(sparse.multiply(sparse.conjugate()).sum(axis=0)).reshape(-1).real

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        output_shape=out_shape,
        input_shape=in_shape,
    )


def _linear_map_from_jax_sparse(
    op: Any,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> LinearMap:
    shape = tuple(int(dim) for dim in op.shape)
    out_shape, in_shape = _map_shapes(shape, input_shape, output_shape)
    dtype = op.dtype
    backend_context = tuple(
        operand
        for operand in (getattr(op, "data", None), getattr(op, "indices", None))
        if operand is not None
    )

    def matvec(vec: Any) -> Any:
        xp = get_array_module(vec, *backend_context)
        return op @ xp.asarray(vec).reshape(shape[1])

    def rmatvec(vec: Any) -> Any:
        xp = get_array_module(vec, *backend_context)
        return op.T @ xp.asarray(vec).reshape(shape[0])

    def matmat(block: Any) -> Any:
        xp = get_array_module(block, *backend_context)
        return op @ xp.asarray(block).reshape(shape[1], -1)

    def rmatmat(block: Any) -> Any:
        xp = get_array_module(block, *backend_context)
        return op.T @ xp.asarray(block).reshape(shape[0], -1)

    def dense_array(xp: Any) -> Any:
        return xp.asarray(op.todense())

    def normal_matrix_diag() -> np.ndarray:
        data = to_numpy(getattr(op, "data", None))
        indices = to_numpy(getattr(op, "indices", None))
        if data.ndim != 1 or indices.ndim != 2 or indices.shape[1] != 2:
            return _normal_matrix_diag_from_matmat(shape, dtype, matmat)
        sparse = scipy.sparse.coo_matrix(
            (data, (indices[:, 0], indices[:, 1])), shape=shape
        ).tocsr()
        return np.asarray(sparse.multiply(sparse.conjugate()).sum(axis=0)).reshape(-1).real

    return LinearMap(
        shape=shape,
        dtype=dtype,
        _matvec=matvec,
        _rmatvec=rmatvec,
        _matmat=matmat,
        _rmatmat=rmatmat,
        _dense_array_func=dense_array,
        _normal_matrix_diag=normal_matrix_diag,
        _backend_context=backend_context,
        output_shape=out_shape,
        input_shape=in_shape,
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

    inferred_input = input_shape or (arr.shape[-1],)
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
        return replace(op, output_shape=out_shape, input_shape=in_shape)

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
        if use_jax():
            try:
                from jax.experimental.sparse import BCOO

                return _linear_map_from_jax_sparse(
                    BCOO.from_scipy_sparse(op), input_shape=input_shape, output_shape=output_shape
                )
            except Exception:
                pass
        return _linear_map_from_scipy_sparse(
            op, input_shape=input_shape, output_shape=output_shape
        )

    try:
        arr = _dense_array_candidate(op)
    except Exception as exc:
        message = f"Unsupported operator type '{type(op)}' for LinearMap conversion."
        raise TypeError(message) from exc
    return _linear_map_from_array(arr, input_shape=input_shape, output_shape=output_shape)
