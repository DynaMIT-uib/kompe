"""Compiled iteration and batching must retain ordinary least-squares meaning."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr as scipy_lsmr

from kompe.math import LinearMap, as_linear_map
from kompe.math.least_squares_problem import LeastSquaresProblem
from kompe.math.least_squares_solver import LeastSquaresSolver

pytestmark = pytest.mark.requires_jax


@pytest.mark.parametrize("algorithm", ["lsmr", "cgls"])
@pytest.mark.parametrize("representation", ["dense", "sparse", "matrix_free", "materialized"])
def test_compiled_iterative_fit_has_independent_batched_rhs(algorithm, representation):
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(16)
    matrix = rng.normal(size=(18, 7))
    device_matrix = jnp.asarray(matrix)
    if representation in {"matrix_free", "materialized"}:
        operator = LinearMap(
            shape=matrix.shape,
            dtype=matrix.dtype,
            matvec=lambda x: device_matrix @ x,
            rmatvec=lambda y: device_matrix.T @ y,
        )
        if representation == "materialized":
            operator.to_matrix(backend="jax")
    else:
        operator = as_linear_map(
            csr_matrix(matrix) if representation == "sparse" else device_matrix
        )
    # Three very different scales, including an exactly zero field.
    rhs = rng.normal(size=(18, 3)) * [1e-8, 0, 1e8]
    penalty = jnp.linspace(0.1, 0.8, 7)
    problem = LeastSquaresProblem(operator, regularization=penalty)
    solver = LeastSquaresSolver(algorithm, tolerance=1e-12)
    with jax.checking_leaks():
        solve = jax.jit(lambda rhs: solver.solve(problem, rhs))
        result = solve(jnp.asarray(rhs))
        repeated = solve(jnp.asarray(2 * rhs))
    expected = np.linalg.lstsq(
        np.vstack([matrix, np.diag(penalty)]), np.vstack([rhs, np.zeros((7, 3))]), rcond=None
    )[0]
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(repeated, 2 * result, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(result[:, 1], 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("case", ["regular", "rank_deficient", "zero_operator", "zero_rhs"])
def test_jitted_lsmr_matches_scipy_with_damping_and_initial_guess(dtype, case):
    import jax
    import jax.numpy as jnp

    from kompe.math.jax_iterative import lsmr

    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(13, 6))
    if np.issubdtype(dtype, np.complexfloating):
        matrix = matrix + 0.3j * rng.normal(size=matrix.shape)
    matrix = matrix.astype(dtype)
    rhs = rng.normal(size=13).astype(dtype)
    if case == "rank_deficient":
        matrix[:, -2:] = matrix[:, :2]
    elif case == "zero_operator":
        matrix[:] = 0
    elif case == "zero_rhs":
        rhs[:] = 0
    initial = rng.normal(size=6).astype(dtype)
    tolerance = (
        1e-6 if np.dtype(dtype).itemsize <= (8 if np.iscomplexobj(matrix) else 4) else 1e-12
    )
    options = {"damp": 0.2, "atol": tolerance, "btol": tolerance, "maxiter": 100, "x0": initial}
    # SciPy's initial min-rho sentinel overflows in single precision;
    # it does not affect the solution used as the independent reference.
    with np.errstate(over="ignore"):
        expected = scipy_lsmr(matrix, rhs, **options)
    operator = as_linear_map(jnp.asarray(matrix))
    actual = jax.jit(lambda b: lsmr(operator, b, **options))(jnp.asarray(rhs))
    np.testing.assert_allclose(actual[0], expected[0], rtol=10 * tolerance, atol=10 * tolerance)
    assert int(actual[1]) == expected[1]


def test_jax_lsmr_reports_exhausted_iterations_inside_jit():
    import jax
    import jax.numpy as jnp

    matrix = np.random.default_rng(9).normal(size=(10, 5))
    problem = LeastSquaresProblem(jnp.asarray(matrix))
    solver = LeastSquaresSolver("lsmr", tolerance=0)
    with pytest.warns(RuntimeWarning, match="stop_code=7"):
        jax.jit(lambda rhs: solver.solve(problem, rhs, maxiter=1))(
            jnp.ones(10)
        ).block_until_ready()


def test_compiled_lsmr_is_reused_for_new_rhs_values(monkeypatch):
    import jax.numpy as jnp

    from kompe.math import jax_iterative

    operator = as_linear_map(jnp.eye(5))
    problem = LeastSquaresProblem(operator)
    solve = LeastSquaresSolver("lsmr").prepare(problem)
    compilations = []
    implementation = jax_iterative.solve_lsmr_columns

    def traced(*args, **kwargs):
        compilations.append(True)
        return implementation(*args, **kwargs)

    monkeypatch.setattr(jax_iterative, "solve_lsmr_columns", traced)
    first = solve(jnp.ones((5, 2)))
    second = solve(jnp.full((5, 2), 3.0, dtype=jnp.float64))
    assert compilations == [True]
    np.testing.assert_allclose(second, 3 * first)


def test_compiled_solve_does_not_keep_discarded_problem_operators_alive():
    import gc
    import weakref

    import jax.numpy as jnp

    operator = as_linear_map(jnp.eye(5))
    reference = weakref.ref(operator)
    problem = LeastSquaresProblem(operator)
    LeastSquaresSolver("lsmr").solve(problem, jnp.ones(5)).block_until_ready()
    del operator, problem
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("method", ["lsmr", "cgls"])
@pytest.mark.parametrize("prepared", [False, True])
def test_constrained_explicit_preconditioner_reuses_compilation(method, prepared, monkeypatch):
    import jax.numpy as jnp

    from kompe.math import jax_iterative

    rng = np.random.default_rng(19)
    problem = LeastSquaresProblem(rng.normal(size=(10, 6)), constraints=rng.normal(size=(1, 6)))
    P = as_linear_map(np.linspace(0.5, 2.0, 6))
    solver = LeastSquaresSolver(method, tolerance=1e-10)
    name = f"solve_{method}_columns"
    implementation = getattr(jax_iterative, name)
    traces = []

    def traced(*args, **kwargs):
        traces.append(1)
        return implementation(*args, **kwargs)

    monkeypatch.setattr(jax_iterative, name, traced)
    solve = solver.prepare(problem, P) if prepared else lambda b: solver.solve(problem, b, P)
    for scale in [1.0, 2.0, 3.0]:
        solve(jnp.full((10, 2), scale)).block_until_ready()
    assert len(traces) == 1


def test_constrained_iterative_fit_can_prepare_inside_jit():
    import jax
    import jax.numpy as jnp

    problem = LeastSquaresProblem(np.eye(4), constraints=np.ones((1, 4)))
    solver = LeastSquaresSolver("lsmr", tolerance=1e-12)
    rhs = jnp.arange(4.0)
    with jax.checking_leaks():
        actual = jax.jit(lambda b: solver.solve(problem, b))(rhs)
    np.testing.assert_allclose(actual, rhs - jnp.mean(rhs), atol=1e-12)


@pytest.mark.parametrize("method", ["lsmr", "cgls"])
def test_public_constrained_preconditioner_matches_the_automatic_one(method):
    import jax.numpy as jnp

    rng = np.random.default_rng(21)
    problem = LeastSquaresProblem(rng.normal(size=(15, 8)), constraints=rng.normal(size=(2, 8)))
    solver = LeastSquaresSolver(method, tolerance=1e-12, preconditioner="jacobi")
    rhs = jnp.asarray(rng.normal(size=(15, 2)))
    P = solver.build_preconditioner(problem)
    assert P.input_shape == problem.solution_shape
    assert P.output_shape == problem.solution_shape
    actual = solver.solve(problem, rhs, preconditioner=P)
    expected = solver.solve(problem, rhs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["lsmr", "cgls"])
def test_explicit_jax_warm_start_selects_the_device_backend(method):
    import jax
    import jax.numpy as jnp

    from kompe.math import backend_context

    with backend_context("numpy"):
        problem = LeastSquaresProblem(np.eye(3))
        actual = LeastSquaresSolver(method).solve(problem, np.ones(3), x0=jnp.zeros(3))
    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, 1.0, atol=1e-14)


@pytest.mark.parametrize("jitted", [False, True])
def test_jax_cgls_warns_when_true_residual_misses_tolerance(jitted):
    import jax
    import jax.numpy as jnp

    A = np.random.default_rng(22).normal(size=(16, 6)) @ np.diag([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    problem = LeastSquaresProblem(A)
    solver = LeastSquaresSolver("cgls", tolerance=1e-12)

    def solve(b):
        return solver.solve(problem, b, maxiter=1)

    if jitted:
        solve = jax.jit(solve)
    with pytest.warns(RuntimeWarning, match="CGLS solver did not converge"):
        solve(jnp.ones((16, 2))).block_until_ready()
