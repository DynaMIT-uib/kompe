"""Configurable solver for ``LeastSquaresProblem`` objects."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any, Final

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_solve, cholesky
from scipy.sparse.linalg import LinearOperator, cg, lsmr, splu

from kompe.math.backend import (
    block_after_jax_linalg,
    block_until_ready,
    get_array_module,
    to_numpy,
)

from .least_squares_problem import LeastSquaresProblem
from .linear_map import LinearMap, as_linear_map, diagonal_linear_map

ITERATION_SAFETY_FACTOR: Final = 10
LEAST_SQUARES_SOLVER_ENV: Final = "KOMPE_LEAST_SQUARES_SOLVER"
LSMR_TOLERANCE_STOP_CODES: Final = frozenset({0, 1, 2})
PreconditionerInput = LinearOperator | LinearMap | None


def _squared_objective_weights(sqrt_weights, size):
    """Return validated diagonal weights for normal equations."""
    if sqrt_weights is None:
        return np.ones(size)
    values = np.asarray(sqrt_weights, dtype=float).reshape(-1)
    if values.size != size:
        raise ValueError(f"sqrt_weights must contain {size} values; got {values.size}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("sqrt_weights must be finite and non-negative.")
    return values**2


def _reshape_columns(values, size, *, array_module=np):
    """View one vector or a block as columns of a fixed height."""
    array = array_module.asarray(values)
    return array.reshape(size) if array.ndim == 1 else array.reshape(size, -1)


def _scale_rows(values, row_weights, *, array_module=np):
    """Apply one-dimensional weights to a vector or column block."""
    weights = array_module.asarray(row_weights)
    return weights * values if values.ndim == 1 else weights.reshape(-1, 1) * values


def dense_full_rank_least_squares_factor(data_matrix, *, sqrt_weights=None) -> np.ndarray:
    """Return the lower Cholesky factor of a weighted normal matrix."""
    data = np.asarray(data_matrix)
    if data.ndim != 2:
        raise ValueError(f"data_matrix must be two-dimensional; got shape {data.shape}.")
    data_size, solution_size = data.shape
    if data_size < solution_size:
        raise ValueError("data_matrix must have at least as many rows as columns.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data_matrix must contain only finite values.")

    objective_weights = _squared_objective_weights(sqrt_weights, data_size)
    data_adjoint = data.T if np.isrealobj(data) else data.T.conjugate()
    if np.all(objective_weights == 1.0):
        normal_matrix = data_adjoint @ data
    else:
        normal_matrix = data_adjoint @ (objective_weights.reshape(-1, 1) * data)
    try:
        return cholesky(normal_matrix, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("data_matrix must have full column rank.") from exc


def _solve_cholesky_factor(factor, rhs, array_module):
    """Solve a positive-definite system from its lower factor."""
    factor_is_complex = np.issubdtype(factor.dtype, np.complexfloating)
    if array_module is not np:
        from jax.scipy.linalg import solve_triangular

        lower_factor = array_module.asarray(factor)
        intermediate = solve_triangular(lower_factor, rhs, lower=True)
        return solve_triangular(
            array_module.swapaxes(array_module.conjugate(lower_factor), -2, -1),
            intermediate,
            lower=False,
        )
    if np.iscomplexobj(rhs) and not factor_is_complex:
        return cho_solve((factor, True), rhs.real, check_finite=False) + 1j * cho_solve(
            (factor, True), rhs.imag, check_finite=False
        )
    return cho_solve((factor, True), rhs, check_finite=False)


def dense_full_rank_least_squares_map(
    data_matrix, *, sqrt_weights=None, input_shape=None, output_shape=None, normal_factor=None
) -> LinearMap:
    """Factor a dense full-column-rank least-squares response map.

    The returned operator maps ``b`` to the unique minimizer of
    ``||W (A x - b)||``. A Cholesky factorization of ``A* W**2 A`` is
    retained, while the rectangular analysis matrix remains implicit.
    """
    data = np.asarray(data_matrix)
    if data.ndim != 2:
        raise ValueError(f"data_matrix must be two-dimensional; got shape {data.shape}.")
    data_size, solution_size = data.shape
    if data_size < solution_size:
        raise ValueError("data_matrix must have at least as many rows as columns.")
    objective_weights = _squared_objective_weights(sqrt_weights, data_size)
    data_adjoint = data.T if np.isrealobj(data) else data.T.conjugate()
    if normal_factor is None:
        factor = dense_full_rank_least_squares_factor(data, sqrt_weights=sqrt_weights)
    else:
        factor = np.asarray(normal_factor)
        expected_shape = (solution_size, solution_size)
        if factor.shape != expected_shape:
            raise ValueError(
                f"normal_factor must have shape {expected_shape}; got {factor.shape}."
            )

    def solve_coefficients(grid_values):
        array_module = get_array_module(grid_values)
        values = _reshape_columns(grid_values, data_size, array_module=array_module)
        weighted_values = _scale_rows(values, objective_weights, array_module=array_module)
        rhs = array_module.asarray(data_adjoint) @ weighted_values
        return _solve_cholesky_factor(factor, rhs, array_module)

    def solve_adjoint(coefficients):
        array_module = get_array_module(coefficients)
        values = _reshape_columns(coefficients, solution_size, array_module=array_module)
        analyzed = array_module.asarray(data) @ _solve_cholesky_factor(
            factor, values, array_module
        )
        return _scale_rows(analyzed, objective_weights, array_module=array_module)

    return LinearMap(
        shape=(solution_size, data_size),
        dtype=np.result_type(data.dtype, objective_weights.dtype),
        _matvec=lambda values: solve_coefficients(values).reshape(-1),
        _rmatvec=lambda values: solve_adjoint(values).reshape(-1),
        _matmat=solve_coefficients,
        _rmatmat=solve_adjoint,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def factorized_least_squares_map(
    data_operator, normal_factor, *, sqrt_weights=None, input_shape=None, output_shape=None
) -> LinearMap:
    """Return analysis from an operator and normal factor."""
    data = as_linear_map(data_operator)
    data_size, solution_size = data.shape
    factor = np.asarray(normal_factor)
    expected_shape = (solution_size, solution_size)
    if factor.shape != expected_shape:
        raise ValueError(f"normal_factor must have shape {expected_shape}; got {factor.shape}.")
    objective_weights = _squared_objective_weights(sqrt_weights, data_size)

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
        _matvec=lambda values: solve_coefficients(values).reshape(-1),
        _rmatvec=lambda values: solve_adjoint(values).reshape(-1),
        _matmat=solve_coefficients,
        _rmatmat=solve_adjoint,
        _backend_context=(*data.backend_context, factor),
        input_shape=input_shape,
        output_shape=output_shape,
    )


def sparse_constrained_least_squares_map(
    data_matrix, constraint_matrix, *, sqrt_weights=None, input_shape=None, output_shape=None
) -> LinearMap:
    """Factor a sparse equality-constrained least-squares response map.

    The returned operator maps ``b`` to the unique constrained
    minimizer of ``||W (A x - b)||`` subject to ``C x = 0``. Its
    adjoint reuses the sparse KKT factorization, so the map remains
    composable without dense materialization.
    """
    data = sp.csr_matrix(data_matrix)
    constraints = sp.csr_matrix(constraint_matrix)
    data_size, solution_size = data.shape
    if constraints.shape[1] != solution_size:
        raise ValueError("constraint_matrix must have the same number of columns as data_matrix.")

    objective_weights = _squared_objective_weights(sqrt_weights, data_size)

    weighted_data = sp.diags(np.sqrt(objective_weights)) @ data
    normal_matrix = weighted_data.T.conjugate() @ weighted_data
    kkt_matrix = sp.bmat(
        [[normal_matrix, constraints.T.conjugate()], [constraints, None]], format="csc"
    )
    factor = splu(kkt_matrix)
    data_adjoint = data.T.conjugate().tocsr()
    data_coo = data.tocoo()
    data_coo_parts = (data_coo.data, np.column_stack([data_coo.row, data_coo.col]), data_coo.shape)
    data_adjoint_coo = data_adjoint.tocoo()
    data_adjoint_coo_parts = (
        data_adjoint_coo.data,
        np.column_stack([data_adjoint_coo.row, data_adjoint_coo.col]),
        data_adjoint_coo.shape,
    )
    constraint_size = constraints.shape[0]
    factor_is_complex = np.issubdtype(factor.L.dtype, np.complexfloating)

    def jax_sparse_matmul(matrix_parts, values):
        from jax.experimental.sparse import BCOO

        array_module = get_array_module(values)
        matrix_data, matrix_indices, matrix_shape = matrix_parts
        operator = BCOO(
            (array_module.asarray(matrix_data), array_module.asarray(matrix_indices)),
            shape=matrix_shape,
        )
        return operator @ values

    def solve_factor_numpy(rhs, *, trans="N"):
        values = np.asarray(rhs)
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

        result_dtype = np.result_type(kkt_matrix.dtype, np.dtype(rhs.dtype))
        result_shape = jax.ShapeDtypeStruct(rhs.shape, result_dtype)

        def callback(values):
            return np.asarray(solve_factor_numpy(values, trans=trans), dtype=result_dtype)

        return jax.pure_callback(callback, result_shape, rhs)

    def append_constraint_zeros(values):
        array_module = get_array_module(values)
        shape = (constraint_size,) if values.ndim == 1 else (constraint_size, values.shape[1])
        return array_module.concatenate(
            [values, array_module.zeros(shape, dtype=values.dtype)], axis=0
        )

    def solve_coefficients(grid_values):
        array_module = get_array_module(grid_values)
        values = _reshape_columns(grid_values, data_size, array_module=array_module)
        weighted_values = _scale_rows(values, objective_weights, array_module=array_module)
        normal_rhs = (
            data_adjoint @ weighted_values
            if array_module is np
            else jax_sparse_matmul(data_adjoint_coo_parts, weighted_values)
        )
        rhs = append_constraint_zeros(normal_rhs)
        return solve_factor(rhs)[:solution_size]

    def solve_adjoint(coefficients):
        array_module = get_array_module(coefficients)
        values = _reshape_columns(coefficients, solution_size, array_module=array_module)
        rhs = append_constraint_zeros(values)
        normal_solution = solve_factor(rhs, trans="H")[:solution_size]
        analyzed = (
            data @ normal_solution
            if array_module is np
            else jax_sparse_matmul(data_coo_parts, normal_solution)
        )
        return _scale_rows(analyzed, objective_weights, array_module=array_module)

    return LinearMap(
        shape=(solution_size, data_size),
        dtype=np.result_type(data.dtype, objective_weights.dtype),
        _matvec=lambda values: solve_coefficients(values).reshape(-1),
        _rmatvec=lambda values: solve_adjoint(values).reshape(-1),
        _matmat=solve_coefficients,
        _rmatmat=solve_adjoint,
        input_shape=input_shape,
        output_shape=output_shape,
    )


class LeastSquaresSolver:
    """A collection of algorithms for solving least-squares problems."""

    VALID_SOLVERS: Final[tuple[str, ...]] = ("normal_solve", "normal_pinv", "lsmr", "cgls", "svd")
    VALID_PRECONDITIONERS: Final[tuple[str, ...]] = ("jacobi", "pinv")

    def __init__(
        self, solver: str = "lsmr", tolerance: float = 1e-15, preconditioner: str | None = None
    ):
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f"Solver must be one of {self.VALID_SOLVERS}")
        self.solver = solver
        self.tolerance = tolerance

        if preconditioner is not None and preconditioner not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        self.preconditioner_type = preconditioner

        self._solve_methods: dict[str, Callable] = {
            "svd": self._solve_svd,
            "normal_solve": self._solve_normal_solve,
            "normal_pinv": self._solve_normal_pinv,
            "lsmr": self._solve_lsmr,
            "cgls": self._solve_cgls,
        }

    def solve(
        self,
        problem: LeastSquaresProblem,
        rhs: np.ndarray | list[np.ndarray],
        preconditioner: PreconditionerInput = None,
        **kwargs,
    ) -> Any:
        """Solve least-squares problem for given right-hand side(s)."""
        preconditioner_map = self._prepare_preconditioner(problem, preconditioner)
        rhs_block, rhs_shape, num_rhs = problem.assemble_rhs_block(
            rhs, include_regularization=self.solver != "normal_pinv"
        )
        if rhs_block is None:
            dtype = problem.A[0].dtype if problem.A else np.float64
            return get_array_module().zeros(problem.solution_shape + rhs_shape, dtype=dtype)

        solver_func = self._solve_methods[self.solver]
        solution_block = solver_func(problem, rhs_block, num_rhs, preconditioner_map, **kwargs)
        return solution_block.reshape(problem.solution_shape + rhs_shape)

    def build_preconditioner(
        self, problem: LeastSquaresProblem, preconditioner_type: str | None = None
    ) -> LinearMap | None:
        """Build preconditioner for the specified solver and problem."""
        selected_type = (
            preconditioner_type if preconditioner_type is not None else self.preconditioner_type
        )
        if selected_type is None:
            return None
        if selected_type not in self.VALID_PRECONDITIONERS:
            raise ValueError(f"Preconditioner must be one of {self.VALID_PRECONDITIONERS}")
        if self.solver == "cgls":
            return self._build_normal_eq_preconditioner(problem, selected_type)
        if self.solver == "lsmr":
            return self._build_lsmr_preconditioner(problem, selected_type)
        return None

    def build_response_solver(
        self, problem: LeastSquaresProblem, preconditioner: PreconditionerInput = None
    ) -> Callable[[np.ndarray | list[np.ndarray]], Any]:
        """Return a reusable solver for matching RHS response blocks."""
        preconditioner_map = self._prepare_preconditioner(problem, preconditioner)
        if self.solver == "normal_pinv":
            return self._build_normal_pinv_response_solver(problem)

        def solve_response(rhs: np.ndarray | list[np.ndarray]) -> Any:
            return self.solve(problem, rhs, preconditioner=preconditioner_map)

        return solve_response

    def _solve_svd(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *_args, **_kwargs
    ) -> np.ndarray:
        xp = get_array_module(rhs_block)
        u, s, vt = problem.svd
        rhs_np = to_numpy(rhs_block)
        cutoff = self.tolerance * (s[0] if s.size > 0 else 0)
        safe_s = np.where(s > cutoff, s, 1.0)
        s_inv = np.where(s > cutoff, 1.0 / safe_s, np.zeros_like(s))
        solution = vt.T.conj() @ (s_inv[:, None] * (u.T.conj() @ rhs_np))
        return xp.asarray(solution)

    def _solve_normal_solve(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *_args, **_kwargs
    ) -> np.ndarray:
        """Solve the normal equations with a direct dense solve."""
        xp, normal_matrix, normal_rhs = self._dense_normal_equations(problem, rhs_block)
        return block_after_jax_linalg(xp.linalg.solve(normal_matrix, normal_rhs))

    def _solve_normal_pinv(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray, *_args, **_kwargs
    ) -> np.ndarray:
        """Solve through the pseudo-inverse of the normal equations."""
        normal_rhs = problem.data_operator.rmatmat(rhs_block)
        normal_pinv = problem.dense_normal_pinv(self.tolerance)
        # Finish this dependent backend matmul before callers assemble
        # NumPy/SciPy blocks.
        return block_until_ready(normal_pinv @ normal_rhs)

    def _build_normal_pinv_response_solver(
        self, problem: LeastSquaresProblem
    ) -> Callable[[np.ndarray | list[np.ndarray]], Any]:
        """Build a normal-pinv response solver with cached factors."""
        normal_pinv = problem.dense_normal_pinv(self.tolerance)
        data_operator = problem.data_operator
        xp = get_array_module(normal_pinv, *data_operator.backend_context)

        def solve_response(rhs: np.ndarray | list[np.ndarray]) -> Any:
            rhs_block, rhs_shape, _ = problem.assemble_rhs_block(rhs, include_regularization=False)
            if rhs_block is None:
                dtype = problem.A[0].dtype if problem.A else np.float64
                return xp.zeros(problem.solution_shape + rhs_shape, dtype=dtype)
            solution_block = normal_pinv @ data_operator.rmatmat(rhs_block)
            return block_until_ready(solution_block.reshape(problem.solution_shape + rhs_shape))

        return solve_response

    def _dense_normal_equations(
        self, problem: LeastSquaresProblem, rhs_block: np.ndarray
    ) -> tuple[Any, Any, Any]:
        """Return dense normal-equation matrix and RHS."""
        xp, _, system_matrix_adjoint, normal_matrix = problem.dense_normal_equations()
        rhs = block_until_ready(xp.asarray(rhs_block))
        return xp, normal_matrix, system_matrix_adjoint @ rhs

    def _solve_lsmr(
        self,
        problem: LeastSquaresProblem,
        rhs_block: np.ndarray,
        num_rhs: int,
        preconditioner: LinearMap | None,
        **kwargs,
    ) -> np.ndarray:
        xp = get_array_module(rhs_block)
        if xp is not np:
            return self._solve_lsmr_jax(problem, rhs_block, num_rhs, preconditioner, **kwargs)

        system_map = problem.system_operator()
        solve_map, recover_solution = self._preconditioned_system(system_map, preconditioner)
        lsmr_options = self._lsmr_options(system_map, kwargs)
        linear_operator = solve_map.as_linear_operator()
        rhs_np = to_numpy(rhs_block)
        columns = []
        for column in range(num_rhs):
            solution_y, stop_code, *_ = lsmr(linear_operator, rhs_np[:, column], **lsmr_options)
            self._warn_if_lsmr_not_converged(stop_code, column)
            columns.append(recover_solution(solution_y))
        return np.column_stack(columns)

    def _solve_lsmr_jax(
        self,
        problem: LeastSquaresProblem,
        rhs_block: Any,
        num_rhs: int,
        preconditioner: LinearMap | None,
        **kwargs,
    ) -> Any:
        """Solve rectangular least squares with internal JAX LSMR."""
        from kompe.math.jax_lsmr import lsmr as jax_lsmr

        xp = get_array_module(rhs_block)
        system_map = problem.system_operator()
        solve_map, recover_solution = self._preconditioned_system(system_map, preconditioner)
        lsmr_options = self._lsmr_options(system_map, kwargs)
        columns = []
        for column in range(num_rhs):
            solution_y, stop_code, *_ = jax_lsmr(solve_map, rhs_block[:, column], **lsmr_options)
            self._warn_if_lsmr_not_converged(stop_code, column)
            columns.append(recover_solution(solution_y))
        return xp.stack(columns, axis=1)

    def _preconditioned_system(
        self, system_map: LinearMap, preconditioner: LinearMap | None
    ) -> tuple[LinearMap, Callable[[Any], Any]]:
        """Return the solve operator and solution transform."""
        if preconditioner is None:
            return system_map, lambda y_vec: y_vec
        return system_map @ preconditioner, preconditioner.matvec

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

        system_map = problem.system_operator()
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
        columns = []
        for column in range(num_rhs):
            sol, exit_code = cg(normal_op, cg_rhs[:, column], **cg_kwargs)
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
        from jax.scipy.sparse.linalg import cg as jax_cg

        system_map = problem.system_operator()
        cg_rhs = system_map.rmatmat(rhs_block).reshape(problem.solution_size, num_rhs)
        max_iter = kwargs.pop("maxiter", ITERATION_SAFETY_FACTOR * problem.solution_size)
        tolerance = kwargs.pop("tol", kwargs.pop("rtol", self.tolerance))
        cg_kwargs = {"tol": tolerance, "atol": kwargs.pop("atol", 0.0), "maxiter": max_iter}
        cg_kwargs.update(kwargs)

        def normal_matvec(x_vec):
            return system_map.rmatvec(system_map.matvec(x_vec))

        apply_preconditioner = None if preconditioner is None else preconditioner.matvec
        columns = []
        for column in range(num_rhs):
            sol, _ = jax_cg(normal_matvec, cg_rhs[:, column], M=apply_preconditioner, **cg_kwargs)
            columns.append(sol)
        return get_array_module(rhs_block).stack(columns, axis=1)

    def _prepare_preconditioner(
        self, problem: LeastSquaresProblem, preconditioner: PreconditionerInput
    ) -> LinearMap | None:
        """Return a validated preconditioner for an iterative solver."""
        if preconditioner is None:
            return None
        if self.solver not in {"lsmr", "cgls"}:
            raise ValueError(f"Solver '{self.solver}' does not accept a preconditioner.")
        preconditioner_map = as_linear_map(preconditioner)
        expected_shape = (problem.solution_size, problem.solution_size)
        if preconditioner_map.shape != expected_shape:
            raise ValueError(
                f"Preconditioner shape {preconditioner_map.shape} != expected {expected_shape}"
            )
        return preconditioner_map

    def _build_normal_eq_preconditioner(
        self, problem: LeastSquaresProblem, preconditioner_type: str
    ) -> LinearMap:
        if preconditioner_type == "jacobi":
            return self._build_jacobi_preconditioner(problem, square_root=False)
        if preconditioner_type == "pinv":
            return self._build_pinv_preconditioner(problem, squared=True)
        raise NotImplementedError(
            f"Preconditioner '{preconditioner_type}' not implemented for CGLS solver."
        )

    def _build_lsmr_preconditioner(
        self, problem: LeastSquaresProblem, preconditioner_type: str
    ) -> LinearMap:
        if preconditioner_type == "jacobi":
            return self._build_jacobi_preconditioner(problem, square_root=True)
        if preconditioner_type == "pinv":
            return self._build_pinv_preconditioner(problem, squared=False)
        raise NotImplementedError(
            f"Preconditioner '{preconditioner_type}' not implemented for LSMR solver."
        )

    def _build_jacobi_preconditioner(
        self, problem: LeastSquaresProblem, *, square_root: bool
    ) -> LinearMap:
        """Build a diagonal preconditioner from ``diag(A* A)``."""
        diag = problem.system_operator().normal_matrix_diag()
        inv_diag = np.divide(1.0, diag, out=np.ones_like(diag), where=diag != 0)
        values = np.sqrt(inv_diag) if square_root else inv_diag
        return diagonal_linear_map(
            values, input_shape=problem.solution_shape, output_shape=problem.solution_shape
        )

    def _build_pinv_preconditioner(
        self, problem: LeastSquaresProblem, *, squared: bool
    ) -> LinearMap:
        """Build a spectral pseudo-inverse preconditioner."""
        vt, s_pinv, s_pinv_sq = self._get_pinv_components(problem, self.tolerance)
        weights = s_pinv_sq if squared else s_pinv
        return self._build_spectral_preconditioner(
            problem.solution_size, vt, weights, problem.solution_shape
        )

    def _build_spectral_preconditioner(
        self, size: int, vt: Any, weights: Any, solution_shape: tuple[int, ...]
    ) -> LinearMap:
        """Build a backend-aware spectral preconditioner."""
        xp = get_array_module(vt, weights)
        vt_arr = xp.asarray(vt)
        weights_arr = xp.asarray(weights)

        def matvec(x_flat):
            x = xp.asarray(x_flat).reshape(size)
            return vt_arr.T.conj() @ (weights_arr * (vt_arr @ x))

        def rmatvec(x_flat):
            x = xp.asarray(x_flat).reshape(size)
            return vt_arr.T.conj() @ (xp.conjugate(weights_arr) * (vt_arr @ x))

        def matmat(block):
            x = xp.asarray(block).reshape(size, -1)
            return vt_arr.T.conj() @ (weights_arr.reshape(-1, 1) * (vt_arr @ x))

        def rmatmat(block):
            x = xp.asarray(block).reshape(size, -1)
            return vt_arr.T.conj() @ (xp.conjugate(weights_arr).reshape(-1, 1) * (vt_arr @ x))

        dtype = np.result_type(vt_arr.dtype, weights_arr.dtype)
        return LinearMap(
            shape=(size, size),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
            _backend_context=(vt_arr, weights_arr),
            output_shape=solution_shape,
            input_shape=solution_shape,
        )

    def _get_pinv_components(
        self, problem: LeastSquaresProblem, tol: float
    ) -> tuple[Any, Any, Any]:
        """Return SVD factors for preconditioners."""
        xp = get_array_module()
        if xp is np:
            _, s, vt = problem.svd
            s_pinv = np.zeros_like(s)
            cutoff = tol * (s[0] if s.size > 0 else 0)
            s_pinv[s > cutoff] = 1.0 / s[s > cutoff]
        else:
            system_matrix = block_until_ready(problem.assemble_dense_system_matrix())
            _, s, vt = block_after_jax_linalg(xp.linalg.svd(system_matrix, full_matrices=False))
            cutoff = tol * (s[0] if s.size > 0 else 0)
            safe_s = xp.where(s > cutoff, s, 1.0)
            s_pinv = xp.where(s > cutoff, 1.0 / safe_s, xp.zeros_like(s))
            s_pinv = block_until_ready(s_pinv)

        return vt, s_pinv, s_pinv**2


def get_default_least_squares_solver(default: str = "normal_pinv") -> str:
    """Return the configured default least-squares solver."""
    solver = os.environ.get(LEAST_SQUARES_SOLVER_ENV, default)
    if solver not in LeastSquaresSolver.VALID_SOLVERS:
        raise ValueError(f"Solver must be one of {LeastSquaresSolver.VALID_SOLVERS}")
    return solver
