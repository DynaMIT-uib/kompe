"""Tests for least-squares solver helpers."""

import warnings

import numpy as np
import pytest
from scipy.sparse.linalg import lsmr as scipy_lsmr

from kompe.math import LinearMap, as_linear_map, get_array_module, jax_enabled, set_backend
from kompe.math.least_squares_problem import LeastSquaresProblem
from kompe.math.least_squares_solver import (
    LEAST_SQUARES_SOLVER_ENV,
    LeastSquaresSolver,
    dense_full_rank_least_squares_map,
    get_default_least_squares_solver,
    sparse_constrained_least_squares_map,
)

# Problem definition and regularization


def test_default_solver_reads_only_the_canonical_environment(monkeypatch):
    """Consumer-specific environment names do not affect Kompe."""
    monkeypatch.delenv(LEAST_SQUARES_SOLVER_ENV, raising=False)
    monkeypatch.setenv("PYNAMIT_LEAST_SQUARES_SOLVER", "lsmr")
    assert get_default_least_squares_solver() == "normal_pinv"

    monkeypatch.setenv(LEAST_SQUARES_SOLVER_ENV, "cgls")
    assert get_default_least_squares_solver() == "cgls"


def test_unregularized_problem_skips_normal_diagonal_scaling():
    """No-reg problems skip a potentially expensive normal diagonal."""
    problem = LeastSquaresProblem(A=np.eye(2), solution_shape=2, data_shapes=2)

    assert problem.regularization_row_scales == []
    assert "data_operator" not in problem.__dict__

    zero_weight_problem = LeastSquaresProblem(
        A=np.eye(2),
        solution_shape=2,
        data_shapes=2,
        regularization_matrices=np.eye(2),
        regularization_strengths=0.0,
    )
    assert zero_weight_problem.regularization_row_scales == [0.0]
    assert "data_operator" not in zero_weight_problem.__dict__


@pytest.mark.parametrize(
    ("regularization_matrix", "regularization_weight"),
    [(np.eye(2), 1e-30), (1e-20 * np.eye(2), 1e-40)],
)
def test_positive_regularization_is_never_silently_discarded(
    regularization_matrix, regularization_weight
):
    """Every positive requested regularization term remains in the objective."""
    problem = LeastSquaresProblem(
        A=np.eye(2),
        solution_shape=2,
        data_shapes=2,
        regularization_matrices=regularization_matrix,
        regularization_strengths=regularization_weight,
    )

    [(scaled_weight, active_matrix)] = problem._active_regularization_terms()
    assert scaled_weight > 0.0
    assert active_matrix is problem.regularization_matrices[0]
    assert problem.system_operator().shape == (4, 2)


def test_positive_strength_on_zero_regularization_operator_is_explicit():
    """A requested no-op regularizer should not look as though it was applied."""
    problem = LeastSquaresProblem(
        A=np.eye(2),
        solution_shape=2,
        data_shapes=2,
        regularization_matrices=np.zeros((2, 2)),
        regularization_strengths=1.0,
    )

    with pytest.raises(ValueError, match="is zero but has positive strength"):
        problem.system_operator()


# Explicit least-squares maps


@pytest.mark.parametrize("complex_data", [False, True])
def test_sparse_constrained_least_squares_map_matches_kkt_and_adjoint(complex_data):
    """Constrained analysis handles rectangular and complex RHS data."""
    A = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
            [1.0, -1.0, 0.0],
            [0.5, 0.25, 1.0],
        ]
    )
    if complex_data:
        A = A + 0.1j * np.array(
            [
                [0.0, 1.0, -0.5],
                [0.5, 0.0, 0.25],
                [-1.0, 0.5, 0.0],
                [0.25, -0.5, 1.0],
            ]
        )
    constraint = np.array([[1.0, 1.0, 1.0]])
    sqrt_weights = np.array([1.0, 2.0, 0.5, 1.5])
    operator = sparse_constrained_least_squares_map(
        A,
        constraint,
        sqrt_weights=sqrt_weights,
        input_shape=(2, 2),
        output_shape=(3,),
    )
    rhs = np.array([1.0 + 0.5j, -0.25j, 2.0 - 0.75j, -1.0 + 0.25j])

    weights = np.diag(sqrt_weights**2)
    kkt = np.block(
        [
            [A.T.conjugate() @ weights @ A, constraint.T],
            [constraint, np.zeros((1, 1))],
        ]
    )
    expected_rhs = np.concatenate([A.T.conjugate() @ weights @ rhs, np.zeros(1)])
    expected = np.linalg.solve(kkt, expected_rhs)[: A.shape[1]]
    analysis_rhs = np.vstack(
        [A.T.conjugate() @ weights, np.zeros((constraint.shape[0], A.shape[0]))]
    )
    analysis_matrix = np.linalg.solve(kkt, analysis_rhs)[: A.shape[1]]

    np.testing.assert_allclose(operator.matvec(rhs), expected)
    np.testing.assert_allclose(constraint @ operator.matvec(rhs), np.zeros(1), atol=1e-14)
    coefficient_probe = np.array([0.5 - 0.25j, 1.0j, -0.75 + 0.1j])
    expected_adjoint = analysis_matrix.T.conjugate() @ coefficient_probe
    np.testing.assert_allclose(operator.rmatvec(coefficient_probe), expected_adjoint)
    np.testing.assert_allclose(
        np.vdot(coefficient_probe, operator.matvec(rhs)),
        np.vdot(operator.rmatvec(coefficient_probe), rhs),
        rtol=1e-13,
        atol=1e-13,
    )

    if jax_enabled():
        import jax
        import jax.numpy as jnp

        compiled = jax.jit(operator.matvec)(jnp.asarray(rhs))
        compiled_adjoint = jax.jit(operator.rmatvec)(jnp.asarray(coefficient_probe))
        np.testing.assert_allclose(compiled, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(compiled_adjoint, expected_adjoint, rtol=1e-13, atol=1e-13)


def test_dense_full_rank_least_squares_map_matches_weighted_lstsq_and_adjoint():
    """Factorized analysis preserves weighted solutions and adjoints."""
    data_matrix = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
            [1.0, -1.0, 0.0],
            [0.5, 0.25, 1.0],
            [-0.5, 0.75, 0.25],
        ]
    )
    sqrt_weights = np.array([1.0, 2.0, 0.5, 1.5, 0.75])
    operator = dense_full_rank_least_squares_map(
        data_matrix, sqrt_weights=sqrt_weights, input_shape=(5,), output_shape=(3,)
    )
    rhs = np.array([1.0 + 0.5j, -0.25j, 2.0 - 0.75j, -1.0 + 0.25j, 0.5j])

    expected = np.linalg.lstsq(
        sqrt_weights.reshape(-1, 1) * data_matrix, sqrt_weights * rhs, rcond=None
    )[0]
    np.testing.assert_allclose(operator.matvec(rhs), expected, rtol=1e-13, atol=1e-13)

    coefficient_probe = np.array([0.5 - 0.25j, 1.0j, -0.75 + 0.1j])
    np.testing.assert_allclose(
        np.vdot(coefficient_probe, operator.matvec(rhs)),
        np.vdot(operator.rmatvec(coefficient_probe), rhs),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.requires_jax
def test_dense_full_rank_least_squares_map_is_jittable_with_jax():
    """Factorized dense analysis preserves the runtime array backend."""
    import jax
    import jax.numpy as jnp

    data_matrix = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0], [0.5, 0.25]])
    operator = dense_full_rank_least_squares_map(data_matrix)
    rhs = jnp.array([1.0, -0.5, 2.0, 0.25])

    actual = jax.jit(operator.matvec)(rhs)
    expected = np.linalg.lstsq(data_matrix, np.asarray(rhs), rcond=None)[0]

    assert "jax" in type(actual).__module__
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


# SciPy LSMR diagnostics


@pytest.mark.parametrize("stop_code", [0, 1, 2])
def test_lsmr_configured_tolerance_stop_codes_do_not_warn(stop_code):
    """LSMR termination at a configured tolerance is quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LeastSquaresSolver._warn_if_lsmr_not_converged(stop_code, column=0)


@pytest.mark.parametrize("stop_code", [4, 5])
def test_lsmr_machine_precision_stop_codes_warn_precisely(stop_code):
    """A machine-precision limit remains visible and precise."""
    with pytest.warns(RuntimeWarning, match="reached machine precision"):
        LeastSquaresSolver._warn_if_lsmr_not_converged(stop_code, column=0)


@pytest.mark.parametrize("stop_code", [3, 6, 7])
def test_lsmr_limit_stop_codes_warn(stop_code):
    """Condition and iteration limit termination remains visible."""
    with pytest.warns(RuntimeWarning, match=rf"stop_code={stop_code}"):
        LeastSquaresSolver._warn_if_lsmr_not_converged(stop_code, column=0)


# Dense solvers and reusable factorizations


def test_normal_pinv_solves_block_rhs():
    """Normal-equation pseudo-inverse supports reusable RHS maps."""
    A = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


def test_normal_pinv_uses_normal_equation_cutoff():
    """Normal pseudo-inverse applies cutoff after forming A* A."""
    A = np.diag([1.0, 1e-8])
    rhs = np.array([1.0, 1e-8])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    np.testing.assert_allclose(solution, np.array([1.0, 0.0]))


def test_normal_pinv_keeps_modes_above_normal_equation_cutoff():
    """Normal pseudo-inverse keeps modes above the A* A cutoff."""
    A = np.diag([1.0, 1e-6])
    rhs = np.array([1.0, 1e-6])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    np.testing.assert_allclose(solution, np.array([1.0, 1.0]))


def test_normal_pinv_does_not_use_direct_solve(monkeypatch):
    """Normal pseudo-inverse also used for full-rank systems."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    def fail_solve(*args, **kwargs):
        raise AssertionError("normal_pinv should apply a pseudo-inverse, not solve")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


def test_normal_pinv_response_solver_reuses_factorization(monkeypatch):
    """Reusable normal-pinv response solves cache dense factors."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs_first = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    rhs_second = np.array([[0.0, 4.0], [2.5, -1.0], [1.5, 3.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
    solve_response = solver.build_response_solver(problem)

    def fail_dense_assembly():
        raise AssertionError("response solver should reuse cached dense factors")

    monkeypatch.setattr(problem, "assemble_dense_system_matrix", fail_dense_assembly)

    A_H = A.T.conj()
    normal_pinv = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True)
    np.testing.assert_allclose(solve_response(rhs_first), normal_pinv @ (A_H @ rhs_first))
    np.testing.assert_allclose(solve_response(rhs_second), normal_pinv @ (A_H @ rhs_second))


def test_unregularized_system_reuses_data_operator():
    """Without regularization there is one canonical system map."""
    problem = LeastSquaresProblem(A=np.eye(3), solution_shape=3, data_shapes=3)

    assert problem.system_operator() is problem.data_operator


def test_normal_pinv_discards_only_derived_regularized_matrix():
    """Keep the data matrix used by repeated solves, not its augmented copy."""
    problem = LeastSquaresProblem(
        A=np.eye(3),
        solution_shape=3,
        data_shapes=3,
        regularization_matrices=np.eye(3),
        regularization_strengths=0.1,
    )
    regularized_system = problem.system_operator()

    problem.dense_normal_pinv(1e-13)

    xp = get_array_module()
    assert regularized_system is not problem.data_operator
    assert regularized_system._cached_dense(xp) is None
    assert problem.data_operator._cached_dense(xp) is not None


def test_least_squares_requires_at_least_one_rhs_term():
    """A missing right-hand side is an input error, not an implicit zero solve."""
    problem = LeastSquaresProblem(A=np.eye(2), solution_shape=2, data_shapes=2)

    with pytest.raises(ValueError, match="At least one right-hand-side"):
        LeastSquaresSolver(solver="normal_pinv").solve(problem, None)

    solve_response = LeastSquaresSolver(solver="normal_pinv").build_response_solver(problem)
    with pytest.raises(ValueError, match="At least one right-hand-side"):
        solve_response(None)


def test_least_squares_rejects_ambiguous_rhs_layout():
    """Equal element counts do not make an unrelated array shape meaningful."""
    problem = LeastSquaresProblem(A=np.eye(6), solution_shape=6, data_shapes=(2, 3))

    with pytest.raises(ValueError, match="incompatible with data_shape"):
        LeastSquaresSolver(solver="normal_pinv").solve(problem, np.ones((3, 4)))


def test_normal_pinv_response_solver_uses_explicit_data_adjoint():
    """Repeated dense response solves do not revisit structured callbacks."""
    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    adjoint_applications = 0

    def rmatmat(values):
        nonlocal adjoint_applications
        adjoint_applications += 1
        return matrix.T @ values

    operator = LinearMap(
        shape=matrix.shape,
        dtype=matrix.dtype,
        _matvec=lambda values: matrix @ values,
        _rmatvec=lambda values: matrix.T @ values,
        _matmat=lambda values: matrix @ values,
        _rmatmat=rmatmat,
        _dense_array_func=lambda xp: xp.asarray(matrix),
    )
    problem = LeastSquaresProblem(A=operator, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
    solve_response = solver.build_response_solver(problem)
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])

    expected = np.linalg.pinv(matrix) @ rhs
    np.testing.assert_allclose(solve_response(rhs), expected)
    np.testing.assert_allclose(solve_response(2 * rhs), 2 * expected)
    assert adjoint_applications == 0


def test_normal_pinv_solve_reuses_cached_pseudo_inverse(monkeypatch):
    """Repeated dense normal-pinv solves reuse the cached n^3 factor."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs_first = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    rhs_second = np.array([[0.0, 4.0], [2.5, -1.0], [1.5, 3.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
    calls = 0
    original_pinv = np.linalg.pinv

    def counted_pinv(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_pinv(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", counted_pinv)

    first = solver.solve(problem, rhs_first)
    second = solver.solve(problem, rhs_second)

    A_H = A.T.conj()
    normal_pinv = original_pinv(A_H @ A, rtol=solver.tolerance, hermitian=True)
    np.testing.assert_allclose(first, normal_pinv @ (A_H @ rhs_first))
    np.testing.assert_allclose(second, normal_pinv @ (A_H @ rhs_second))
    assert len(problem._dense_normal_pinv_cache) == 1
    assert calls == (0 if jax_enabled() else 1)


# Dense JAX solves


@pytest.mark.requires_jax
@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv"])
def test_dense_solvers_preserve_jax_output_when_backend_enabled(solver_name):
    """Dense solvers preserve JAX output when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_matrix = problem.assemble_dense_system_matrix()
        assert "jax" in type(rhs_block).__module__
        assert "jax" in type(system_matrix).__module__
        solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.requires_jax
def test_uncached_normal_pinv_stays_on_jax(monkeypatch):
    """An in-memory JAX factorization does not cross through NumPy."""
    import kompe.math.least_squares_problem as problem_module

    problem = LeastSquaresProblem(
        A=np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]]),
        solution_shape=2,
        data_shapes=3,
    )
    previous_backend = jax_enabled()

    def reject_host_transfer(_array):
        raise AssertionError("uncached JAX pseudo-inverse crossed to NumPy")

    monkeypatch.setattr(problem_module, "to_numpy", reject_host_transfer)
    try:
        set_backend("jax")
        normal_pinv = problem.dense_normal_pinv(1e-13)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(normal_pinv).__module__


@pytest.mark.requires_jax
def test_svd_solver_preserves_jax_output_when_backend_enabled():
    """SVD solver keeps JAX-facing assembly and output."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_matrix = problem.assemble_dense_system_matrix()
        assert "jax" in type(rhs_block).__module__
        assert "jax" in type(system_matrix).__module__
        solver = LeastSquaresSolver(solver="svd", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.requires_jax
def test_least_squares_problem_follows_jax_operator_context_when_numpy_active():
    """JAX-backed operator terms should drive matrix-free assembly."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])

    try:
        set_backend("numpy")
        problem = LeastSquaresProblem(A=jnp.asarray(A), solution_shape=2, data_shapes=3)
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_block = problem.system_operator().matmat(np.eye(2))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(rhs_block).__module__
    assert "jax" in type(system_block).__module__
    np.testing.assert_allclose(np.asarray(system_block), A)


@pytest.mark.requires_jax
def test_normal_pinv_matches_numpy_hermitian_reference_when_jax_enabled():
    """JAX normal-pinv matches the hermitian reference."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


# Iterative solvers


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_solver_solves_block_rhs_with_base_preconditioner(solver_name):
    """Iterative block RHS solves reuse the base preconditioner."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0, -1.0], [3.0, 1.0, 0.5], [0.5, -2.0, 4.0], [1.5, 0.0, 2.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-12, preconditioner="jacobi")
    preconditioner = solver.build_preconditioner(problem)

    assert preconditioner.shape == (2, 2)
    solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_solvers_do_not_materialize_dense_system(monkeypatch, solver_name):
    """Iterative solves stay matrix-free for no dense preconditioner."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-12)

    def fail_dense_assembly():
        raise AssertionError("iterative solvers should not assemble dense systems")

    monkeypatch.setattr(problem, "assemble_dense_system_matrix", fail_dense_assembly)

    solution = solver.solve(problem, rhs, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_jacobi_preconditioner_does_not_materialize_dense_system(
    monkeypatch, solver_name
):
    """Jacobi-preconditioned iterative solves stay matrix-free."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-12, preconditioner="jacobi")

    def fail_dense_assembly():
        raise AssertionError(
            "jacobi-preconditioned iterative solvers should not assemble dense systems"
        )

    monkeypatch.setattr(problem, "assemble_dense_system_matrix", fail_dense_assembly)

    preconditioner = solver.build_preconditioner(problem)
    solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.requires_jax
@pytest.mark.parametrize("solver_name", ["cgls", "lsmr"])
@pytest.mark.parametrize("preconditioner_type", [None, "jacobi", "pinv"])
def test_iterative_solvers_preserve_jax_output_when_backend_enabled(
    solver_name, preconditioner_type
):
    """Iterative solvers preserve JAX output when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(
            solver=solver_name, tolerance=1e-12, preconditioner=preconditioner_type
        )
        preconditioner = solver.build_preconditioner(problem)
        solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.requires_jax
def test_jax_lsmr_solves_underdetermined_block_rhs():
    """Internal JAX LSMR handles rectangular underdetermined systems."""
    A = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=4, data_shapes=3)
    previous_backend = jax_enabled()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(solver="lsmr", tolerance=1e-12)
        solution = solver.solve(problem, rhs, maxiter=200)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.requires_jax
@pytest.mark.parametrize("complex_system", [False, True])
def test_jax_lsmr_recurrence_matches_scipy(complex_system):
    """Internal LSMR matches SciPy for damping and an initial guess."""
    import jax.numpy as jnp

    from kompe.math.jax_lsmr import lsmr as jax_lsmr

    rng = np.random.default_rng(2841)
    matrix = rng.normal(size=(8, 5))
    rhs = rng.normal(size=8)
    initial_guess = rng.normal(size=5)
    if complex_system:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        rhs = rhs + 1j * rng.normal(size=rhs.shape)
        initial_guess = initial_guess + 1j * rng.normal(size=initial_guess.shape)

    options = {
        "damp": 0.25,
        "atol": 1e-12,
        "btol": 1e-12,
        "conlim": 1e10,
        "maxiter": 100,
        "x0": initial_guess,
    }
    expected = scipy_lsmr(matrix, rhs, **options)
    actual = jax_lsmr(as_linear_map(jnp.asarray(matrix)), jnp.asarray(rhs), **options)

    assert actual[1:3] == expected[1:3]
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual[3:], expected[3:], rtol=1e-11, atol=1e-12)


@pytest.mark.requires_jax
def test_jax_lsmr_zero_rhs_discards_non_solution_initial_guess():
    """A zero RHS rejects an initial guess that is not a solution."""
    import jax.numpy as jnp

    from kompe.math.jax_lsmr import lsmr as jax_lsmr

    matrix = np.array([[2.0, -1.0], [1.0, 3.0], [0.5, 2.0]])
    rhs = np.zeros(3)
    initial_guess = np.array([1.0, -2.0])

    expected = scipy_lsmr(matrix, rhs, x0=initial_guess)
    actual = jax_lsmr(as_linear_map(jnp.asarray(matrix)), jnp.asarray(rhs), x0=initial_guess)

    assert actual[1:3] == expected[1:3]
    np.testing.assert_allclose(actual[0], expected[0])


@pytest.mark.requires_jax
def test_jax_lsmr_uses_complex_operator_dtype():
    """A complex operator promotes real right-hand sides correctly."""
    import jax.numpy as jnp

    from kompe.math.jax_lsmr import lsmr as jax_lsmr

    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    rhs = rng.normal(size=8)

    solution, stop_code, *_ = jax_lsmr(
        as_linear_map(jnp.asarray(matrix)), jnp.asarray(rhs), atol=1e-12, btol=1e-12, maxiter=100
    )

    assert stop_code in {1, 2}
    assert np.issubdtype(solution.dtype, np.complexfloating)
    expected = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


# Public solver validation


@pytest.mark.parametrize("weight", [-1.0, np.inf, np.nan, np.array([1.0, 2.0])])
def test_regularization_strengths_must_be_finite_non_negative_scalars(weight):
    """Invalid regularization weights fail before system assembly."""
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        LeastSquaresProblem(
            A=np.eye(2),
            solution_shape=2,
            data_shapes=2,
            regularization_matrices=np.eye(2),
            regularization_strengths=weight,
        )


def test_regularization_does_not_mutate_a_custom_data_normal_matrix():
    """A supplied normal matrix remains owned by its builder."""
    data_normal = np.eye(2)
    problem = LeastSquaresProblem(
        A=np.eye(2),
        solution_shape=2,
        data_shapes=2,
        regularization_matrices=np.eye(2),
        regularization_strengths=1.0,
        data_normal_matrix_builder=lambda: data_normal,
    )

    np.testing.assert_allclose(problem.dense_normal_matrix(), 2.0 * np.eye(2))
    np.testing.assert_array_equal(data_normal, np.eye(2))


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_solver_tolerance_must_be_a_finite_non_negative_scalar(tolerance):
    """Reject solver tolerances that cannot define a numerical cutoff."""
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        LeastSquaresSolver(tolerance=tolerance)


def test_solver_tolerance_must_be_scalar_numeric_data():
    """Reject booleans explicitly and let array conversion report its own error."""
    with pytest.raises(TypeError, match="finite non-negative scalar"):
        LeastSquaresSolver(tolerance=True)
    with pytest.raises(TypeError):
        LeastSquaresSolver(tolerance=np.array([1.0, 2.0]))


@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd"])
@pytest.mark.parametrize("entrypoint", ["solve", "build_response_solver"])
def test_dense_solvers_reject_explicit_preconditioners(solver_name, entrypoint):
    """Dense solvers reject explicitly supplied preconditioners."""
    problem = LeastSquaresProblem(A=np.eye(2), solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver=solver_name)
    preconditioner = as_linear_map(np.eye(2))

    with pytest.raises(ValueError, match="does not accept a preconditioner"):
        if entrypoint == "solve":
            solver.solve(problem, np.ones(2), preconditioner=preconditioner)
        else:
            solver.build_response_solver(problem, preconditioner=preconditioner)


@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd"])
def test_dense_solvers_do_not_build_configured_preconditioners(solver_name):
    """A requested preconditioner is never silently ignored."""
    problem = LeastSquaresProblem(A=np.eye(2), solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver=solver_name, preconditioner="jacobi")

    with pytest.raises(ValueError, match="does not accept a preconditioner"):
        solver.build_preconditioner(problem)
