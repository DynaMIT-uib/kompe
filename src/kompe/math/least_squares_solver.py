"""Configurable solver for ``LeastSquaresProblem`` objects."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, Final

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_solve, cholesky, lu_solve
from scipy.sparse.linalg import LinearOperator, cg, lsmr, splu

from kompe.cache import BoundedCache
from kompe.math.backend import (
    _is_jax_tracer,
    block_until_ready,
    get_array_module,
    get_backend,
    synchronize_linalg_result,
    to_numpy,
)

from .least_squares_problem import LeastSquaresProblem, as_rhs_block
from .linear_map import (
    LinearMap,
    _normalized_constraint_rows,
    as_linear_map,
    diagonal_linear_map,
    vstack_linear_maps,
)

ITERATION_SAFETY_FACTOR: Final = 10
LEAST_SQUARES_SOLVER_ENV: Final = "KOMPE_LEAST_SQUARES_SOLVER"
LSMR_TOLERANCE_STOP_CODES: Final = frozenset({0, 1, 2})
PreconditionerInput = LinearOperator | LinearMap | None


def _inverse_singular_values(singular_values, tolerance):
    """Invert retained singular values and leave the numerical nullspace zero."""
    xp = get_array_module(singular_values)
    cutoff = tolerance * (singular_values[0] if singular_values.size else 0.0)
    retained = singular_values > cutoff
    # Excluded zeros must not be divided by, even in JAX's unselected branch.
    denominator = xp.where(retained, singular_values, 1.0)
    return xp.where(retained, 1.0 / denominator, 0.0)


def _residual_weights(sqrt_weights, size, *, dtype=float):
    """Return validated diagonal residual weights."""
    if sqrt_weights is None:
        return np.ones(size, dtype=dtype)
    values = np.asarray(sqrt_weights, dtype=dtype).reshape(-1)
    if values.size != size:
        raise ValueError(f"sqrt_weights must contain {size} values; got {values.size}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("sqrt_weights must be finite and non-negative.")
    return values


def _reshape_columns(values, size, *, array_module=np):
    """View one vector or a block as columns of a fixed height."""
    array = array_module.asarray(values)
    return array.reshape(size) if array.ndim == 1 else array.reshape(size, -1)


def _scale_rows(values, row_weights, *, array_module=np):
    """Apply one-dimensional weights to a vector or column block."""
    weights = array_module.asarray(row_weights)
    return weights * values if values.ndim == 1 else weights.reshape(-1, 1) * values


def dense_full_rank_least_squares_map(
    data_matrix, *, sqrt_weights=None, input_shape=None, output_shape=None
) -> LinearMap:
    """Build full-rank analysis from a dense synthesis matrix on the CPU.

    Return the map ``b -> argmin_x ||W (A x - b)||``. Retain the
    lower Cholesky factor of ``A* W**2 A`` instead of a rectangular
    analysis matrix. For an existing factor and a structured synthesis
    operator, use :func:`cholesky_least_squares_map` directly.
    """
    data = np.asarray(data_matrix)
    if data.ndim != 2:
        raise ValueError(f"data_matrix must be two-dimensional; got shape {data.shape}.")
    data_size, solution_size = data.shape
    if data_size < solution_size:
        raise ValueError("data_matrix must have at least as many rows as columns.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data_matrix must contain only finite values.")

    objective_weights = _residual_weights(sqrt_weights, data_size) ** 2
    data_adjoint = data.T if np.isrealobj(data) else data.T.conjugate()
    if np.all(objective_weights == 1.0):
        normal_matrix = data_adjoint @ data
    else:
        normal_matrix = data_adjoint @ (objective_weights.reshape(-1, 1) * data)
    try:
        factor = cholesky(normal_matrix, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("data_matrix must have full column rank.") from exc
    return cholesky_least_squares_map(
        as_linear_map(data, input_shape=output_shape, output_shape=input_shape),
        factor,
        sqrt_weights=sqrt_weights,
    )


def _solve_cholesky_factor(factor, rhs, array_module):
    """Solve a positive-definite system from its lower factor."""
    if array_module is not np:
        from jax.scipy.linalg import cho_solve as solve

        return solve((factor, True), rhs)
    if np.iscomplexobj(rhs) and not np.iscomplexobj(factor):
        # Keep the real LAPACK factor: promoting the whole matrix to
        # complex for every RHS costs more than these two real solves.
        return cho_solve((factor, True), rhs.real, check_finite=False) + 1j * cho_solve(
            (factor, True), rhs.imag, check_finite=False
        )
    return cho_solve((factor, True), rhs, check_finite=False)


def cholesky_least_squares_map(data_operator, normal_factor, *, sqrt_weights=None) -> LinearMap:
    """Return analysis from a synthesis operator and lower normal factor.

    The analysis domain and codomain are the synthesis operator's
    output and input shapes. Reading these axes does not materialize it.
    """
    data = as_linear_map(data_operator)
    data_size, solution_size = data.shape
    factor = np.asarray(normal_factor)
    expected_shape = (solution_size, solution_size)
    if factor.shape != expected_shape:
        raise ValueError(f"normal_factor must have shape {expected_shape}; got {factor.shape}.")
    objective_weights = _residual_weights(sqrt_weights, data_size) ** 2

    def solve_coefficients(grid_values):
        array_module = data.array_module(grid_values, factor)
        values = _reshape_columns(grid_values, data_size, array_module=array_module)
        weighted_values = _scale_rows(values, objective_weights, array_module=array_module)
        rhs = data.rmatmat(weighted_values)
        return _solve_cholesky_factor(factor, rhs, array_module)

    def solve_adjoint(coefficients):
        array_module = data.array_module(coefficients, factor)
        values = _reshape_columns(coefficients, solution_size, array_module=array_module)
        analyzed = data.matmat(_solve_cholesky_factor(factor, values, array_module))
        return _scale_rows(analyzed, objective_weights, array_module=array_module)

    return LinearMap(
        shape=(solution_size, data_size),
        dtype=np.result_type(data.dtype, factor.dtype, objective_weights.dtype),
        matvec=lambda values: solve_coefficients(values).reshape(-1),
        rmatvec=lambda values: solve_adjoint(values).reshape(-1),
        matmat=solve_coefficients,
        rmatmat=solve_adjoint,
        backend_operands=(*data.backend_operands, factor),
        input_shape=data.output_shape,
        output_shape=data.input_shape,
    )


def sparse_least_squares_map(
    data_matrix,
    constraint_matrix=None,
    *,
    regularization=None,
    sqrt_weights=None,
    input_shape=None,
    output_shape=None,
) -> LinearMap:
    """Factor a sparse equality-constrained least-squares response map.

    The returned operator maps ``b`` to the unique constrained
    minimizer of ``||W (A x - b)||² + ||R x||²`` subject to ``C x = 0``.
    Both the already-scaled penalty R and the constraints C are optional. Its
    adjoint reuses the sparse KKT factorization, so the map remains
    composable without dense materialization.

    Factorization and solves use SciPy on the CPU. JAX application
    uses a host callback for each RHS block; it is JIT-compatible,
    not a GPU-native sparse solve.
    """
    data = sp.csr_matrix(data_matrix)
    data_size, solution_size = data.shape
    constraints = (
        sp.csr_matrix((0, solution_size), dtype=data.dtype)
        if constraint_matrix is None
        else sp.csr_matrix(constraint_matrix)
    )
    if constraints.shape[1] != solution_size:
        raise ValueError("constraint_matrix must have the same number of columns as data_matrix.")
    constraints = _normalized_constraint_rows(constraints)
    penalty = (
        sp.csr_matrix((0, solution_size), dtype=data.dtype)
        if regularization is None
        else sp.csr_matrix(regularization)
    )
    if penalty.shape[1] != solution_size:
        raise ValueError("regularization must have the same number of columns as data_matrix.")

    dtype = np.result_type(
        data.dtype,
        constraints.dtype,
        penalty.dtype,
        0.0 if sqrt_weights is None else np.asarray(sqrt_weights).dtype,
        0.0,
    )
    real_dtype = np.empty((), dtype=dtype).real.dtype
    residual_weights = _residual_weights(sqrt_weights, data_size, dtype=real_dtype)

    # A common residual scale does not change this constrained minimizer.
    # Remove it before forming the KKT system, whose constraint rows have
    # an independent scale. Otherwise units alone can spoil its pivots.
    weighted_data = sp.diags(residual_weights) @ data
    residual_scale = max(
        np.max(np.abs(weighted_data.data), initial=0.0),
        np.max(np.abs(penalty.data), initial=0.0),
    )
    if residual_scale > 0:
        # Sparse true-division promotes float32 in SciPy. Multiplication
        # preserves precision, while normalizing W b separately avoids
        # squaring a potentially very small/large residual scale.
        weighted_data = weighted_data * np.reciprocal(residual_scale)
        penalty = penalty * np.reciprocal(residual_scale)
        residual_weights = residual_weights / residual_scale
    normal_matrix = weighted_data.T.conjugate() @ weighted_data + penalty.T.conjugate() @ penalty
    kkt_matrix = sp.bmat(
        [[normal_matrix, constraints.T.conjugate()], [constraints, None]], format="csc"
    )
    factors = BoundedCache(2)
    factors.store(kkt_matrix.dtype, splu(kkt_matrix))
    data_operator = as_linear_map(weighted_data)
    constraint_size = constraints.shape[0]

    def solve_factor_numpy(rhs, *, trans="N"):
        values = np.asarray(rhs)
        # A later higher-precision RHS needs matching factors, just as for
        # dense normal_solve. Complex RHS can still use two real solves.
        dtype = np.result_type(kkt_matrix.dtype, values.real.dtype)
        factor = factors.get_or_create(dtype, lambda: splu(kkt_matrix.astype(dtype)))
        factor_is_complex = np.issubdtype(dtype, np.complexfloating)
        if np.iscomplexobj(values) and not factor_is_complex:
            return factor.solve(values.real, trans=trans) + 1j * factor.solve(
                values.imag, trans=trans
            )
        return factor.solve(values, trans=trans)

    def solve_factor(rhs, *, trans="N"):
        array_module = get_array_module(rhs)
        if array_module is np:
            return solve_factor_numpy(rhs, trans=trans)

        import jax

        result_dtype = jax.dtypes.canonicalize_dtype(
            np.result_type(kkt_matrix.dtype, np.dtype(rhs.dtype))
        )
        result_shape = jax.ShapeDtypeStruct(rhs.shape, result_dtype)

        def callback(values):
            return np.asarray(solve_factor_numpy(values, trans=trans), dtype=result_dtype)

        return jax.pure_callback(callback, result_shape, rhs, vmap_method="sequential")

    def append_constraint_zeros(values):
        array_module = get_array_module(values)
        shape = (constraint_size,) if values.ndim == 1 else (constraint_size, values.shape[1])
        return array_module.concatenate(
            [values, array_module.zeros(shape, dtype=values.dtype)], axis=0
        )

    def solve_coefficients(grid_values):
        array_module = get_array_module(grid_values)
        values = _reshape_columns(grid_values, data_size, array_module=array_module)
        weighted_values = _scale_rows(values, residual_weights, array_module=array_module)
        normal_rhs = data_operator.rmatmat(weighted_values)
        rhs = append_constraint_zeros(normal_rhs)
        return solve_factor(rhs)[:solution_size]

    def solve_adjoint(coefficients):
        array_module = get_array_module(coefficients)
        values = _reshape_columns(coefficients, solution_size, array_module=array_module)
        rhs = append_constraint_zeros(values)
        normal_solution = solve_factor(rhs, trans="H")[:solution_size]
        analyzed = data_operator.matmat(normal_solution)
        return _scale_rows(analyzed, residual_weights, array_module=array_module)

    return LinearMap(
        shape=(solution_size, data_size),
        dtype=kkt_matrix.dtype,
        matvec=lambda values: solve_coefficients(values).reshape(-1),
        rmatvec=lambda values: solve_adjoint(values).reshape(-1),
        matmat=solve_coefficients,
        rmatmat=solve_adjoint,
        input_shape=input_shape,
        output_shape=output_shape,
    )


class LeastSquaresSolver:
    """A selected least-squares algorithm with reusable problem factors.

    An omitted ``method`` uses ``KOMPE_LEAST_SQUARES_SOLVER`` (default
    ``normal_pinv``). The algorithm is fixed when this object is created.

    ``tolerance`` defaults to ``1e-15``. For ``svd``, it is a
    relative cutoff on singular values of the
    weighted, regularized system. For ``normal_pinv``, it is a cutoff
    on the normal matrix's eigenvalues (squared singular values).
    LSMR and CGLS use it as their iterative convergence tolerance.
    ``normal_solve`` is a direct LU solve and does not truncate modes.
    """

    VALID_SOLVERS: Final[tuple[str, ...]] = ("normal_solve", "normal_pinv", "lsmr", "cgls", "svd")
    VALID_PRECONDITIONERS: Final[tuple[str, ...]] = ("jacobi", "pinv")

    def __init__(
        self,
        method: str | None = None,
        tolerance: float = 1e-15,
        preconditioner: str | None = None,
    ):
        if method is None:
            method = get_default_least_squares_solver()
        elif method not in self.VALID_SOLVERS:
            raise ValueError(f"Solver must be one of {self.VALID_SOLVERS}")
        self.method = method
        if isinstance(tolerance, (bool, np.bool_)):
            raise TypeError("tolerance must be a finite non-negative scalar.")
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be a finite non-negative scalar.")
        self.tolerance = tolerance

        if preconditioner is not None and preconditioner not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        self.preconditioner_type = preconditioner

    def solve(
        self,
        problem: LeastSquaresProblem,
        rhs: np.ndarray | list[np.ndarray],
        preconditioner: PreconditionerInput = None,
        **kwargs,
    ) -> Any:
        """Solve least squares with algorithm-specific keyword options.

        Nonzero LSMR ``damp`` cannot be combined with a right preconditioner,
        which would change the regularization coordinates. Express that
        regularization in ``problem`` instead.
        ``x0`` is in the original coefficient space: one field (or flat
        vector) shared by all RHSs, or ``solution_shape + rhs_batch_shape``.
        LSMR solves for a correction to x0; penalties in the problem retain
        their original zero-centred meaning. Its ``damp`` penalizes the correction.
        """
        original_shape = problem.solution_shape
        problem, solution_basis, preconditioner_map = self._prepare_coordinates(
            problem, preconditioner
        )
        if (
            self.method == "lsmr"
            and preconditioner_map is not None
            and kwargs.get("damp", 0.0) != 0.0
        ):
            raise ValueError(
                "Nonzero LSMR damp cannot be combined with a right preconditioner: "
                "it would penalize preconditioned coordinates. "
                "Specify regularization in LeastSquaresProblem instead."
            )
        rhs_block, rhs_shape, num_rhs = problem.assemble_rhs_block(
            rhs, include_regularization=self.method not in {"normal_pinv", "normal_solve"}
        )
        if kwargs.get("x0") is not None:
            initial, batch_shape = as_rhs_block(kwargs["x0"], original_shape)
            if batch_shape and batch_shape != rhs_shape:
                raise ValueError("x0 must be one coefficient field or have the RHS batch shape.")
            xp = get_array_module(rhs_block, initial)
            rhs_block = xp.asarray(rhs_block)
            initial = xp.broadcast_to(xp.asarray(initial), (initial.shape[0], num_rhs))
            if solution_basis is not None:
                initial = solution_basis.rmatmat(initial)
            kwargs = {**kwargs, "x0": initial}

        if self.method == "svd":
            solver_func = self._solve_svd
        elif self.method == "normal_solve":
            solver_func = self._solve_normal_solve
        elif self.method == "normal_pinv":
            solver_func = self._solve_normal_pinv
        elif self.method == "lsmr":
            solver_func = self._solve_lsmr
        else:
            solver_func = self._solve_cgls
        solution_block = solver_func(problem, rhs_block, num_rhs, preconditioner_map, **kwargs)
        solution = solution_block.reshape(problem.solution_shape + rhs_shape)
        return solution if solution_basis is None else solution_basis(solution)

    def _prepare_coordinates(self, problem, preconditioner):
        """Reuse constrained coordinates and preconditioners for solve and prepare."""
        basis = None
        if problem.constraints is not None and not (
            self.method == "normal_solve" and problem.system_operator.is_sparse
        ):
            basis = problem.solution_basis
            if preconditioner is not None:
                full = self._prepare_preconditioner(problem, preconditioner)
                preconditioner = problem._restricted_preconditioners.get_or_create(
                    full, lambda: basis.adjoint() @ full @ basis
                )
            problem = problem.reduced_problem
        return problem, basis, self._prepare_preconditioner(problem, preconditioner)

    def build_preconditioner(
        self, problem: LeastSquaresProblem, preconditioner_type: str | None = None
    ) -> LinearMap | None:
        """Build and reuse a preconditioner for this algorithm and problem."""
        selected_type = (
            preconditioner_type if preconditioner_type is not None else self.preconditioner_type
        )
        if selected_type is None:
            return None
        if selected_type not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        if self.method not in {"cgls", "lsmr"}:
            raise ValueError(f"Solver '{self.method}' does not accept a preconditioner.")
        key = (
            self.method,
            selected_type,
            self.tolerance if selected_type == "pinv" else None,
            get_backend(*problem.backend_operands),
        )
        if problem.constraints is not None:

            def constrained_preconditioner():
                reduced = self.build_preconditioner(problem.reduced_problem, selected_type)
                Z = problem.solution_basis
                full = Z @ reduced @ Z.adjoint()
                # Public preconditioners act on full coefficient fields;
                # the solve can reuse their original independent coordinates.
                problem._restricted_preconditioners.store(full, reduced)
                return full

            return problem._preconditioner_cache.get_or_create(key, constrained_preconditioner)
        return problem._preconditioner_cache.get_or_create(
            key,
            lambda: (
                self._build_jacobi_preconditioner(problem, square_root=self.method == "lsmr")
                if selected_type == "jacobi"
                else self._build_pinv_preconditioner(problem, squared=self.method == "cgls")
            ),
        )

    def prepare(
        self, problem: LeastSquaresProblem, preconditioner: PreconditionerInput = None
    ) -> Callable[[np.ndarray | list[np.ndarray]], Any]:
        """Return a reusable solver for matching RHS response blocks."""
        problem, basis, preconditioner_map = self._prepare_coordinates(problem, preconditioner)
        if self.method == "normal_pinv":
            solve = self._build_normal_pinv_response_solver(problem)
        else:
            solve = partial(self.solve, problem, preconditioner=preconditioner_map)
        return solve if basis is None else lambda rhs: basis(solve(rhs))

    def _solve_svd(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *_args
    ) -> np.ndarray:
        u, s, vt = problem.svd(backend=get_backend(rhs_block))
        s_inv = _inverse_singular_values(s, self.tolerance)
        return vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ rhs_block))

    def _solve_normal_solve(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *_args
    ) -> np.ndarray:
        """Apply a reusable LU solve to the data term's normal RHS."""
        xp = get_array_module(rhs_block)
        data = problem.data_operator
        if (
            problem.constraints is None
            and data.is_diagonal
            and all(penalty.is_diagonal for penalty in problem.regularization_operators)
        ):
            # Independent coefficients need only vector division, on-device.
            values = data.diagonal()
            penalties = problem.regularization_operators
            if not penalties:
                if xp is np and np.any(values == 0):
                    raise np.linalg.LinAlgError("Singular normal matrix.")
                return rhs_block / values[:, None]
            penalty_values = [penalty.diagonal() for penalty in penalties]
            scale = xp.abs(values)
            for penalty in penalty_values:
                scale = xp.maximum(scale, xp.abs(penalty))
            if xp is np and np.any(scale == 0):
                raise np.linalg.LinAlgError("Singular normal matrix.")
            normalized = values / scale
            diagonal = xp.abs(normalized) ** 2
            for penalty in penalty_values:
                diagonal = diagonal + xp.abs(penalty / scale) ** 2
            return (normalized.conj() / diagonal)[:, None] * (rhs_block / scale[:, None])
        if problem.system_operator.is_sparse:
            if problem._sparse_analysis_operator is None:
                penalty = (
                    vstack_linear_maps(problem.regularization_operators).to_sparse_matrix()
                    if problem.regularization_operators
                    else None
                )
                problem._sparse_analysis_operator = sparse_least_squares_map(
                    data.to_sparse_matrix(),
                    problem.constraints,
                    regularization=penalty,
                    input_shape=data.output_shape,
                    output_shape=problem.solution_shape,
                )
            return problem._sparse_analysis_operator.matmat(rhs_block)
        factors = problem._dense_normal_lu(rhs_block.dtype, backend=get_backend(rhs_block))
        # A higher-precision regularizer also promotes the data adjoint
        # product, just as it does in the full augmented system.
        rhs_block = xp.asarray(block_until_ready(rhs_block), dtype=factors[0].dtype)
        normal_rhs = problem.data_operator.rmatmat(rhs_block)
        solve = lu_solve
        if xp is not np:
            from jax.scipy.linalg import lu_solve as solve

        return synchronize_linalg_result(solve(factors, normal_rhs, check_finite=False))

    def _solve_normal_pinv(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *_args
    ) -> np.ndarray:
        """Solve through the pseudo-inverse of the normal equations."""
        normal_rhs = problem.data_operator.rmatmat(rhs_block)
        normal_pinv = problem.dense_normal_pinv(self.tolerance, backend=get_backend(normal_rhs))
        # Finish this dependent backend matmul before callers assemble
        # NumPy/SciPy blocks.
        return block_until_ready(normal_pinv @ normal_rhs)

    def _build_normal_pinv_response_solver(
        self, problem: LeastSquaresProblem
    ) -> Callable[[np.ndarray | list[np.ndarray]], Any]:
        """Build a dense normal-pinv solver for repeated response blocks."""
        # Retain the prepared factorization independently of cache eviction.
        # LinearMap reuses its device copies when later RHS backends differ.
        normal_pinv = as_linear_map(problem.dense_normal_pinv(self.tolerance))
        data_adjoint = problem.data_operator.adjoint()

        def solve_response(rhs: np.ndarray | list[np.ndarray]) -> Any:
            rhs_block, rhs_shape, _ = problem.assemble_rhs_block(rhs, include_regularization=False)
            backend = get_backend(rhs_block)
            solution_block = normal_pinv.to_matrix(backend=backend) @ (
                data_adjoint.matmat(rhs_block)
            )
            return block_until_ready(solution_block.reshape(problem.solution_shape + rhs_shape))

        return solve_response

    def _solve_lsmr(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_rhs: int,
        preconditioner: LinearMap | None,
        **kwargs,
    ) -> np.ndarray:
        xp = get_array_module(rhs_block)
        system_map = problem.system_operator
        lsmr_options = self._lsmr_options(system_map, kwargs)
        if xp is not np:
            import jax

            from kompe.math.jax_iterative import solve_lsmr_columns

            # The problem owns its compiled kernels. A module-global JIT
            # cache with static map arguments would retain old operators.
            solve = problem._compiled_iterative_solvers.get_or_create(
                ("lsmr", preconditioner),
                lambda: jax.jit(
                    partial(solve_lsmr_columns, system_map, preconditioner=preconditioner),
                    static_argnames=("maxiter",),
                ),
            )
            solution, codes = solve(rhs_block, **lsmr_options)

            def warn(codes):
                for column, code in enumerate(codes):
                    self._warn_if_lsmr_not_converged(int(code), column)

            if _is_jax_tracer(codes):
                # Diagnostics cross to Python only on a failed convergence
                # test, never once per iteration or for successful JIT solves.
                jax.lax.cond(
                    xp.any(codes > 2), lambda: jax.debug.callback(warn, codes), lambda: None
                )
            else:
                warn(to_numpy(codes))
            return solution

        initial = lsmr_options.pop("x0", None)
        if initial is not None:
            # Solve A P y = b - A x0, then x = x0 + P y. The initial
            # guess remains in coefficient space, without inverting P.
            rhs_block = rhs_block - system_map.matmat(initial)
        solve_map = system_map if preconditioner is None else system_map @ preconditioner

        columns = []
        for column in range(num_rhs):
            rhs = rhs_block[:, column]
            # LSMR's initial condition estimate contains unit-sized terms.
            # Scale the entire objective (including damping), not its relative
            # weights, using the first bidiagonalization norm ||A* b||/||b||.
            # This needs one adjoint action, not a matrix or a norm estimate
            # assembled by probing every coefficient.
            norm_rhs = float(xp.linalg.norm(rhs))
            scale = (
                float(xp.linalg.norm(solve_map.rmatvec(rhs / norm_rhs))) if norm_rhs > 0 else 0.0
            )
            if scale == 0.0:
                scale = 1.0  # Zero/orthogonal RHS: let LSMR report its usual stop code.
            # This scale is transient and RHS-specific. Scale the vector
            # actions, not a dense copy of A for every right-hand side.
            normalized_map = LinearMap(
                shape=solve_map.shape,
                dtype=solve_map.dtype,
                matvec=lambda x, scale=scale: solve_map.matvec(x) / scale,
                rmatvec=lambda x, scale=scale: solve_map.rmatvec(x) / scale,
                backend_operands=solve_map.backend_operands,
            )
            options = {**lsmr_options, "damp": lsmr_options.get("damp", 0.0) / scale}
            solution_y, stop_code, *_ = lsmr(
                normalized_map.as_linear_operator(), rhs / scale, **options
            )
            self._warn_if_lsmr_not_converged(stop_code, column)
            columns.append(solution_y)
        solution = xp.stack(columns, axis=1)
        if preconditioner is not None:
            solution = preconditioner.matmat(solution)
        return solution if initial is None else initial + solution

    def _lsmr_options(self, system_map: LinearMap, options: dict[str, Any]) -> dict[str, Any]:
        """Return LSMR options with the default iteration cap."""
        m, n = system_map.shape
        default_max_iterations = ITERATION_SAFETY_FACTOR * min(m, n) if m > 0 and n > 0 else n
        return {
            "atol": self.tolerance,
            "btol": self.tolerance,
            "maxiter": default_max_iterations,
            **options,
        }

    @staticmethod
    def _warn_if_lsmr_not_converged(stop_code: int, column: int) -> None:
        """Warn when LSMR misses a tolerance or numerical limit."""
        if stop_code in LSMR_TOLERANCE_STOP_CODES:
            return
        if stop_code in {4, 5}:
            message = (
                f"LSMR reached machine precision before satisfying the configured tolerances "
                f"for RHS column {column} (stop_code={stop_code})."
            )
        else:
            message = (
                f"LSMR may not have converged for RHS column {column} (stop_code={stop_code})."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _solve_cgls(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_rhs: int,
        preconditioner: LinearMap | None,
        **kwargs,
    ) -> np.ndarray:
        xp = get_array_module(rhs_block)
        if xp is not np:
            return self._solve_cgls_jax(problem, rhs_block, num_rhs, preconditioner, **kwargs)

        system_map = problem.system_operator
        normal_op = LinearOperator(
            (system_map.shape[1], system_map.shape[1]),
            matvec=lambda x: np.asarray(system_map.rmatvec(system_map.matvec(x))),
            dtype=system_map.dtype,
        )
        rhs_np = to_numpy(rhs_block)
        cg_rhs = np.asarray(system_map.rmatmat(rhs_np)).reshape(problem.solution_size, num_rhs)

        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        cg_kwargs = {
            "rtol": self.tolerance,
            "M": preconditioner.as_linear_operator() if preconditioner is not None else None,
            "maxiter": max_iter,
            **kwargs,
        }
        initial = cg_kwargs.pop("x0", None)
        columns = []
        for column in range(num_rhs):
            sol, exit_code = cg(
                normal_op,
                cg_rhs[:, column],
                x0=None if initial is None else initial[:, column],
                **cg_kwargs,
            )
            if exit_code != 0:
                warnings.warn(
                    f"CGLS solver did not converge for RHS column {column} "
                    f"(exit_code={exit_code}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            columns.append(sol)
        return np.column_stack(columns)

    def _solve_cgls_jax(
        self,
        problem: LeastSquaresProblem,
        rhs_block: Any,
        num_rhs: int,
        preconditioner: LinearMap | None,
        **kwargs,
    ) -> Any:
        """Solve normal equations with JAX CG."""
        import jax

        from kompe.math.jax_iterative import solve_cgls_columns

        system_map = problem.system_operator
        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        tolerance = kwargs.pop("tol", kwargs.pop("rtol", self.tolerance))
        cg_kwargs = {"tol": tolerance, "atol": kwargs.pop("atol", 0.0), "maxiter": max_iter}
        cg_kwargs.update(kwargs)

        solve = problem._compiled_iterative_solvers.get_or_create(
            ("cgls", preconditioner),
            lambda: jax.jit(
                partial(solve_cgls_columns, system_map, preconditioner=preconditioner),
                static_argnames=("maxiter",),
            ),
        )
        solution, converged = solve(rhs_block, **cg_kwargs)

        def warn(converged):
            columns = np.flatnonzero(~converged)
            if columns.size:
                warnings.warn(
                    f"CGLS solver did not converge for RHS columns {columns.tolist()}: "
                    "the true normal residual exceeds the requested tolerance.",
                    RuntimeWarning,
                    stacklevel=3,
                )

        if _is_jax_tracer(converged):
            jax.lax.cond(
                get_array_module(converged).all(converged),
                lambda: None,
                lambda: jax.debug.callback(warn, converged),
            )
        else:
            warn(to_numpy(converged))
        return solution

    def _prepare_preconditioner(
        self, problem: LeastSquaresProblem, preconditioner: PreconditionerInput
    ) -> LinearMap | None:
        """Return a validated preconditioner for an iterative solver."""
        if preconditioner is None:
            return self.build_preconditioner(problem)
        if self.method not in {"lsmr", "cgls"}:
            raise ValueError(f"Solver '{self.method}' does not accept a preconditioner.")
        preconditioner_map = as_linear_map(preconditioner)
        expected_shape = (problem.solution_size, problem.solution_size)
        if preconditioner_map.shape != expected_shape:
            raise ValueError(
                f"Preconditioner shape {preconditioner_map.shape} != expected {expected_shape}"
            )
        return preconditioner_map

    def _build_jacobi_preconditioner(
        self, problem: LeastSquaresProblem, *, square_root: bool
    ) -> LinearMap:
        """Build a diagonal preconditioner from ``diag(A* A)``."""
        diag = problem.system_operator.normal_matrix_diag()
        inv_diag = np.divide(1.0, diag, out=np.ones_like(diag), where=diag != 0)
        values = np.sqrt(inv_diag) if square_root else inv_diag
        return diagonal_linear_map(
            values, input_shape=problem.solution_shape, output_shape=problem.solution_shape
        )

    def _build_pinv_preconditioner(
        self, problem: LeastSquaresProblem, *, squared: bool
    ) -> LinearMap:
        """Build a spectral pseudo-inverse preconditioner."""
        _, s, vt = problem.svd()
        s_pinv = _inverse_singular_values(s, self.tolerance)
        weights = s_pinv**2 if squared else s_pinv

        def matvec(values):
            xp = get_array_module(values, vt, weights)
            vectors = xp.asarray(vt)
            values = xp.asarray(values).reshape(problem.solution_size)
            return vectors.T.conj() @ (xp.asarray(weights) * (vectors @ values))

        def matmat(values):
            xp = get_array_module(values, vt, weights)
            vectors = xp.asarray(vt)
            values = xp.asarray(values).reshape(problem.solution_size, -1)
            return vectors.T.conj() @ (xp.asarray(weights)[:, None] * (vectors @ values))

        # Real spectral weights make V diag(weights) V* self-adjoint.
        # Keep this direct kernel: composing three maps adds measurable
        # dispatch overhead to each NumPy preconditioner application.
        return LinearMap(
            shape=(problem.solution_size, problem.solution_size),
            dtype=np.result_type(vt.dtype, weights.dtype),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            rmatmat=matmat,
            backend_operands=(vt, weights),
            input_shape=problem.solution_shape,
            output_shape=problem.solution_shape,
        )


def get_default_least_squares_solver(default: str = "normal_pinv") -> str:
    """Return the configured default least-squares solver."""
    solver = os.environ.get(LEAST_SQUARES_SOLVER_ENV, default)
    if solver not in LeastSquaresSolver.VALID_SOLVERS:
        raise ValueError(f"Solver must be one of {LeastSquaresSolver.VALID_SOLVERS}")
    return solver
