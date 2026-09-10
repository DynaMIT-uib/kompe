"""Least-squares problem definition."""

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext
from functools import cached_property
from typing import Any, TypeAlias

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse.linalg import LinearOperator

from kompe.cache import BoundedCache
from kompe.math.backend import (
    _is_jax_tracer,
    get_array_module,
    immutable_array,
    readonly_numpy_array,
    synchronize_linalg_result,
    to_numpy,
)
from kompe.math.fingerprints import array_fingerprint
from kompe.math.linear_map import (
    LinearMap,
    _normalized_constraint_rows,
    as_linear_map,
    null_space_linear_map,
    vstack_linear_maps,
)

OperatorInput: TypeAlias = np.ndarray | scipy.sparse.spmatrix | LinearOperator | LinearMap
OperatorInputList: TypeAlias = OperatorInput | list[OperatorInput]
_NORMAL_PINV_CACHE_VERSION = 2


def as_rhs_block(values: Any, data_shape: tuple[int, ...]) -> tuple[Any, tuple[int, ...]]:
    """Flatten data axes into rows and batch axes into RHS columns.

    Accept one field, a flat vector, or ``data_shape + batch_shape``.
    Data axes always lead; batch axes always trail, as with ``LinearMap``.
    Return the column block and the original batch shape; solutions use
    ``solution_shape + batch_shape`` regardless of the input layout.
    """
    xp = get_array_module(values)
    array = xp.asarray(values)
    data_size = math.prod(data_shape)
    data_ndim = len(data_shape)

    if array.shape == data_shape:
        return array.reshape(data_size, 1), ()
    if array.ndim > data_ndim and array.shape[:data_ndim] == data_shape:
        batch_shape = array.shape[data_ndim:]
        return array.reshape(data_size, math.prod(batch_shape)), batch_shape
    if array.ndim <= 1 and array.size == data_size:
        return array.reshape(data_size, 1), ()
    raise ValueError(f"Shape {array.shape} incompatible with data_shape {data_shape}.")


def _weight_operator(w_val: Any, shape: tuple[int, ...]) -> LinearMap | None:
    """Normalize residual weights to a shaped linear map."""
    if w_val is None:
        return None
    flat_dim = math.prod(shape)
    if not isinstance(w_val, (LinearMap, LinearOperator)) and not scipy.sparse.issparse(w_val):
        w_val = immutable_array(w_val)
        if w_val.shape == shape:
            w_val = w_val.reshape(flat_dim)
    return as_linear_map(w_val, output_shape=shape, input_shape=shape)


def relative_regularization(A, L, strength, *, sqrt_weights=None):
    """Return R = s L for a dimensionless relative regularization strength.

    Choose s² = strength * median_positive(diag(A* W* W A)) /
    median_positive(diag(L* L)). Supply the same data operator and residual
    weights as the fit. For several data terms, stack their weighted maps
    first. Zero/None strength disables the penalty and returns None.

    This explicit setup step inspects normal diagonals on the CPU, never
    constructs a normal matrix, and preserves diagonal or matrix-free L.
    For a physically specified scale, pass the scaled L straight to
    LeastSquaresProblem instead. Balance before restricting solution space.
    """
    if strength is None:
        return None
    try:
        array = np.asarray(strength)
        if array.ndim != 0:
            raise ValueError
        strength = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError("strength must be a finite non-negative scalar.") from exc
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be a finite non-negative scalar.")
    if strength == 0:
        return None
    if L is None:
        raise ValueError("A positive relative strength requires a regularization operator.")
    A = as_linear_map(A)
    L = as_linear_map(L, input_shape=A.input_shape)
    weights = _weight_operator(sqrt_weights, A.output_shape)
    data = A if weights is None else weights @ A
    data_diagonal = data.normal_matrix_diag()
    positive_data = data_diagonal[data_diagonal > 0]
    if positive_data.size == 0:
        raise ValueError("Relative regularization requires a nonzero weighted data operator.")
    penalty_diagonal = L.normal_matrix_diag()
    positive_penalty = penalty_diagonal[penalty_diagonal > 0]
    if positive_penalty.size == 0:
        raise ValueError("Relative regularization requires a nonzero regularization operator.")
    scale = math.sqrt(strength) * math.sqrt(np.median(positive_data) / np.median(positive_penalty))
    return scale * L


class LeastSquaresProblem:
    """Data fit and optional regularization in one coefficient space.

    With one data term and one regularization term, the solved objective is

    ``||W (A x - b)||² + ||R x||²``.

    ``sqrt_weights`` supplies the diagonal of ``W`` (so its square gives the
    statistical or area weight). ``regularization`` is one already-scaled
    operator R or a list of such operators. No scale is inferred or balanced
    inside the problem. For a dimensionless relative strength, construct R
    explicitly with ``relative_regularization`` before defining the problem.

    Shapes come from the data operators' declared input and output shapes.
    All data operators must share one input shape. Use ``as_linear_map``
    to label the scientific axes of raw arrays before building the problem.

    ``constraints=C`` imposes the exact homogeneous equations ``C x = 0``.
    Sparse direct solves retain these rows in a KKT system; the other methods
    use orthonormal null-space coordinates, preserving spectral cutoffs.
    Use ``restrict_solution(Z)`` for an explicit change of coordinates ``x = Z y``.

    A problem describes a fixed objective: construct a new problem to change
    its operators or weights. Array-valued weights are owned and read-only;
    supplied operators must retain their mathematical action while in use.

    Normal products belong to the data and penalty operators. Iterative
    solvers apply the original system without constructing a normal matrix.
    """

    def __init__(
        self,
        A: OperatorInputList,
        *,
        sqrt_weights: Any | list[Any] | None = None,
        regularization: OperatorInputList | None = None,
        constraints: Any | None = None,
        operator_cache: Any | None = None,
        cache_identity: Any | None = None,
    ):
        self._dense_normal_pinv_cache = BoundedCache(2)
        self._dense_normal_lu_cache = BoundedCache(2)
        self._svd_cache = BoundedCache(2)
        self._preconditioner_cache = BoundedCache(2)
        self._restricted_preconditioners = BoundedCache(2)
        self._compiled_iterative_solvers = BoundedCache(2)
        self._sparse_analysis_operator = None
        self.operator_cache = operator_cache
        self.cache_identity = cache_identity

        self._process_data_terms(A, sqrt_weights)
        if scipy.sparse.issparse(constraints):
            self.constraints = constraints.tocsr(copy=True)
            constraint_values = self.constraints.data
        else:
            self.constraints = None if constraints is None else readonly_numpy_array(constraints)
            constraint_values = self.constraints
        if self.constraints is not None and (
            self.constraints.ndim != 2
            or self.constraints.shape[1] != self.solution_size
            or not np.all(np.isfinite(constraint_values))
        ):
            raise ValueError("constraints must be finite rows with one column per solution entry.")
        self.regularization_operators = [
            as_linear_map(operator, input_shape=self.solution_shape)
            for operator in self._prepare_input_list(
                regularization, "regularization", is_optional=True
            )
        ]

    @cached_property
    def solution_basis(self):
        """Orthonormal coordinates satisfying the declared constraints."""
        if self.constraints is None:
            return None
        rows = self.constraints
        if scipy.sparse.issparse(rows):
            rows = rows.toarray()  # Orthonormal null-space setup is a dense CPU QR.
        return null_space_linear_map(rows, output_shape=self.solution_shape)

    @cached_property
    def reduced_problem(self):
        """The same objective in independent constrained coordinates."""
        if self.constraints is None:
            return self
        identity = None
        if self.cache_identity is not None:
            rows = self.constraints
            signature = (
                (
                    rows.shape,
                    *(array_fingerprint(a) for a in (rows.data, rows.indices, rows.indptr)),
                )
                if scipy.sparse.issparse(rows)
                else array_fingerprint(rows)
            )
            identity = {
                "problem": self.cache_identity,
                "constraints": signature,
            }
        return self.restrict_solution(self.solution_basis, cache_identity=identity)

    def restrict_solution(self, solution_basis, *, cache_identity=None):
        """Return the same objective in coordinates ``x = solution_basis(y)``.

        Solve the returned problem and apply ``solution_basis`` to recover
        x. Regularization keeps its scale in the original coefficient space;
        a constraint must not rebalance the physical objective. An orthonormal
        basis preserves Euclidean norms and spectral-cutoff conventions.
        Persistent caching requires an identity that also identifies the basis.
        Existing constraints become C Z; rows eliminated by this change are
        removed at floating-point roundoff, independently of the fit tolerance.
        Coordinate columns are scaled using Z's normal diagonal for this
        rank test; a matrix-free map can supply it without column probes.
        """
        basis = as_linear_map(solution_basis)
        if basis.output_shape != self.solution_shape:
            raise ValueError(
                "solution_basis output_shape must match the problem's solution_shape."
            )

        # C Z may lose rows when Z already satisfies some or all constraints.
        # Determine its row space against the multiplication's roundoff scale,
        # not against its own tiny residual or the user's fit tolerance.
        constraints = None
        if self.constraints is not None:
            # The objective and coordinate map are fixed setup data, even
            # when the first RHS arrives under JIT. Never cache their tracers.
            setup = nullcontext()
            if get_array_module(*basis.backend_operands) is not np:
                from jax import ensure_compile_time_eval

                setup = ensure_compile_time_eval()
            with setup:
                rows = _normalized_constraint_rows(self.constraints)
                if scipy.sparse.issparse(rows):
                    rows = rows.toarray()
                projected = to_numpy(basis.rmatmat(rows.T.conj())).T.conj()
                column_norms = np.sqrt(basis.normal_matrix_diag())
                column_scales = np.where(column_norms > 0, column_norms, 1)
                # Test in unit-column coordinates so a large unrelated
                # column cannot erase a real constraint on a smaller one.
                _, singular_values, row_space = scipy.linalg.svd(
                    projected / column_scales, full_matrices=False
                )
                precision = max(
                    np.finfo(np.empty((), dtype=dtype).real.dtype).eps
                    for dtype in (rows.dtype, np.result_type(basis.dtype, 0.0))
                )
                cutoff = precision * max(*basis.shape, rows.shape[0])
                retained = singular_values > cutoff
                if np.any(retained):
                    constraints = row_space[retained] * column_scales

        return LeastSquaresProblem(
            [operator @ basis for operator in self.data_operators],
            sqrt_weights=self.weight_operators,
            regularization=[operator @ basis for operator in self.regularization_operators],
            constraints=constraints,
            operator_cache=self.operator_cache,
            cache_identity=cache_identity,
        )

    def _process_data_terms(self, A_in, sqrt_weights_in):
        A_list = self._prepare_input_list(A_in, "A")
        if not A_list:
            raise ValueError("At least one data operator is required.")
        self.data_operators = [as_linear_map(op) for op in A_list]
        self.solution_shape = self.data_operators[0].input_shape
        if any(op.input_shape != self.solution_shape for op in self.data_operators):
            raise ValueError(
                "Data operators must share input_shape; "
                "use as_linear_map to label their axes explicitly."
            )
        self.solution_size = math.prod(self.solution_shape)
        self.data_shapes = [op.output_shape for op in self.data_operators]
        sqrt_weights_list = self._prepare_input_list(
            sqrt_weights_in, "sqrt_weights", count=len(A_list)
        )
        self.weight_operators = [
            _weight_operator(w, self.data_shapes[i]) for i, w in enumerate(sqrt_weights_list)
        ]

    @cached_property
    def backend_operands(self) -> tuple[Any, ...]:
        """Return active operands without assembling the system."""
        operators = self.data_operators + self.weight_operators + self.regularization_operators
        return tuple(
            operand
            for operator in operators
            if operator is not None
            for operand in operator.backend_operands
        )

    @cached_property
    def data_operator(self) -> LinearMap:
        """Assemble the data operator without regularization."""
        row_maps = [
            operator if weight is None else weight @ operator
            for operator, weight in zip(self.data_operators, self.weight_operators, strict=True)
        ]
        return vstack_linear_maps(row_maps, input_shape=self.solution_shape)

    def system_matrix(self, *, backend=None) -> Any:
        """Materialize the regularized system on the requested backend."""
        return self.system_operator.to_matrix(backend=backend)

    def dense_normal_pinv(self, tolerance: float, *, backend=None) -> Any:
        """Return the backend-specific cached normal-matrix pseudo-inverse."""
        xp = get_array_module(*self.backend_operands, backend=backend)
        key = (xp, float(tolerance))
        cached = self._dense_normal_pinv_cache.get(key)
        if cached is not None:
            return cached

        def compute():
            normal_matrix = self.dense_normal_matrix(backend=backend)
            normal_pinv = synchronize_linalg_result(
                xp.linalg.pinv(normal_matrix, rtol=tolerance, hermitian=True)
            )
            if _is_jax_tracer(normal_pinv):
                return normal_pinv
            self._clear_normal_matrix_work()
            return normal_pinv

        def build():
            if self.operator_cache is None or self.cache_identity is None:
                return compute()

            cached = self.operator_cache.get_or_create(
                "least_squares_normal_pinv",
                {
                    "algorithm": "least_squares_normal_pinv",
                    "version": _NORMAL_PINV_CACHE_VERSION,
                    "problem": self.cache_identity,
                    "backend": xp.__name__,
                    "tolerance": float(tolerance),
                },
                lambda: to_numpy(compute()),
            )
            return xp.asarray(cached)

        normal_pinv = build()
        if not _is_jax_tracer(normal_pinv):
            self._dense_normal_pinv_cache.store(key, normal_pinv)
        return normal_pinv

    def _dense_normal_lu(self, rhs_dtype, *, backend=None):
        """Cache direct-solve factors with the RHS's precision included.

        Like ``linalg.solve``, promote the normal matrix and RHS to
        their common inexact dtype before factorization. A later
        higher-precision or complex RHS needs its own factorization.
        """
        xp = get_array_module(*self.backend_operands, backend=backend)
        key = (xp, np.dtype(rhs_dtype))
        cached = self._dense_normal_lu_cache.get(key)
        if cached is not None:
            return cached

        normal = self.dense_normal_matrix(backend=backend)
        normal = xp.asarray(normal, dtype=xp.result_type(normal.dtype, rhs_dtype, 0.0))
        if xp is np:
            # NumPy solve raises on exact singularity. Preserve that
            # contract instead of retaining SciPy's invalid factor.
            with warnings.catch_warnings():
                warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
                try:
                    factors = scipy.linalg.lu_factor(normal, check_finite=False)
                except scipy.linalg.LinAlgWarning as exc:
                    raise np.linalg.LinAlgError("Singular matrix") from exc
        else:
            from jax.scipy.linalg import lu_factor

            factors = synchronize_linalg_result(lu_factor(normal))
        if not _is_jax_tracer(factors[0]):
            self._dense_normal_lu_cache.store(key, factors)
            self._clear_normal_matrix_work()
        return factors

    def _clear_normal_matrix_work(self):
        """Release construction arrays after retaining a factorization.

        Keep the data map: repeated solves still apply its adjoint.
        Discard any separately materialized augmented system.
        """
        regularized_system = self.system_operator
        if regularized_system is not self.data_operator:
            regularized_system.clear_dense_cache()

    def dense_normal_matrix(self, *, backend=None) -> Any:
        """Materialize A* W* W A + sum(L* L) through operator structure."""
        return self.system_operator.normal_operator().to_matrix(backend=backend)

    def svd(self, *, backend=None) -> tuple[Any, Any, Any]:
        """Return cached reduced SVD factors on the requested backend."""
        xp = get_array_module(*self.backend_operands, backend=backend)
        cached = self._svd_cache.get(xp)
        if cached is not None:
            return cached
        matrix = self.system_matrix(backend=backend)
        factors = synchronize_linalg_result(xp.linalg.svd(matrix, full_matrices=False))
        if not _is_jax_tracer(factors[1]):
            self._svd_cache.store(xp, factors)
        return factors

    def assemble_rhs_block(
        self, b: Any | list[Any], *, include_regularization: bool = True
    ) -> tuple[Any, tuple[int, ...], int]:
        """Assemble one or more right-hand side columns."""
        b_list = self._prepare_input_list(b, "b", count=len(self.data_operators))
        processed = [
            (None, None) if b_val is None else as_rhs_block(b_val, self.data_shapes[i])
            for i, b_val in enumerate(b_list)
        ]
        valid_b = [p for p in processed if p[0] is not None]
        if not valid_b:
            raise ValueError("At least one right-hand-side data term must be provided.")
        rhs_shape = valid_b[0][1]
        if not all(p[1] == rhs_shape for p in valid_b):
            raise ValueError("Inconsistent RHS column shapes in b terms.")

        num_rhs = math.prod(rhs_shape) if rhs_shape else 1
        active_regularization_terms = (
            self.regularization_operators if include_regularization else ()
        )
        system_operator = self.system_operator if include_regularization else self.data_operator
        dtype = system_operator.dtype
        # Backend choice uses all active terms, but must not assemble a
        # regularized system just to apply a previously cached inverse.
        xp = get_array_module(*(p[0] for p in valid_b), *self.backend_operands)

        blocks = []
        for i, (b_col_block, _) in enumerate(processed):
            num_a_rows = self.data_operators[i].shape[0]
            if b_col_block is None:
                blocks.append(xp.zeros((num_a_rows, num_rhs), dtype=dtype))
                continue
            w_item = self.weight_operators[i]
            if w_item is not None:
                b_col_block = w_item.matmat(b_col_block)
            blocks.append(xp.asarray(b_col_block).reshape(num_a_rows, num_rhs))

        for L_item in active_regularization_terms:
            blocks.append(xp.zeros((L_item.shape[0], num_rhs), dtype=dtype))

        d_block = xp.asarray(
            xp.vstack(blocks), dtype=xp.result_type(dtype, *(b.dtype for b in blocks))
        )
        return d_block, rhs_shape, num_rhs

    @cached_property
    def system_operator(self) -> LinearMap:
        """Stack regularization rows below the data operator."""
        regularization_terms = self.regularization_operators
        if not regularization_terms:
            return self.data_operator
        row_maps = [self.data_operator]
        row_maps.extend(regularization_terms)
        return vstack_linear_maps(row_maps, input_shape=self.solution_shape)

    @staticmethod
    def _prepare_input_list(
        item: Any | None,
        name: str,
        count: int | None = None,
        is_optional: bool = False,
    ) -> list:
        if item is None:
            if is_optional:
                return []
            if count is None:
                raise ValueError(f"Input '{name}' cannot be None.")
            return [None] * count
        lst = item if isinstance(item, list) else [item]
        if count is not None and len(lst) != count:
            raise ValueError(f"Input '{name}' has {len(lst)} items, but expected {count}.")
        return lst
