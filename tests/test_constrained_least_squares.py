"""Exact constraints and sparse regularization preserve one fit objective."""

import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.sparse import csr_matrix

from kompe import GlobalCSBasis, SHBasis, SphericalTransform
from kompe.math import LinearMap, as_linear_map, get_array_module
from kompe.math.least_squares_problem import LeastSquaresProblem
from kompe.math.least_squares_solver import LeastSquaresSolver, sparse_least_squares_map


@pytest.mark.parametrize("method", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("sparse", [False, True])
def test_constrained_regularized_fit_matches_independent_coordinates(method, sparse):
    rng = np.random.default_rng(6)
    A = rng.normal(size=(14, 9)) + 1j * rng.normal(size=(14, 9))
    R = rng.normal(size=(5, 9)) + 1j * rng.normal(size=(5, 9))
    C = rng.normal(size=(2, 9)) + 1j * rng.normal(size=(2, 9))
    weights = np.linspace(0.4, 2.0, 14)
    rhs = rng.normal(size=(14, 3))
    Z = null_space(C)
    system = np.vstack([weights[:, None] * A, R]) @ Z
    augmented_rhs = np.vstack([weights[:, None] * rhs, np.zeros((5, 3))])
    expected = Z @ np.linalg.lstsq(system, augmented_rhs, rcond=None)[0]
    problem = LeastSquaresProblem(
        csr_matrix(A) if sparse else A,
        sqrt_weights=weights,
        regularization=csr_matrix(R) if sparse else R,
        constraints=csr_matrix(C) if sparse else C,
    )
    solver = LeastSquaresSolver(method, tolerance=1e-12)
    for solve in (lambda b: solver.solve(problem, b), solver.prepare(problem)):
        actual = solve(rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(C @ actual, 0, atol=1e-10)
    if sparse and method == "normal_solve":
        assert "solution_basis" not in problem.__dict__
        assert problem.system_operator.materialized_matrix is None


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_sparse_regularized_inverse_has_the_correct_adjoint_and_relative_scale(scale):
    rng = np.random.default_rng(4)
    A = rng.normal(size=(8, 5))
    R = rng.normal(size=(3, 5)) * (1 + 0.3j)
    C = rng.normal(size=(1, 5))
    weights = np.linspace(0.2, 2, 8)
    rhs = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
    probe = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2))
    Z = null_space(C)
    inverse = Z @ np.linalg.pinv(np.vstack([weights[:, None] * A, R]) @ Z)[:, :8] * weights
    operator = sparse_least_squares_map(
        A, C, sqrt_weights=scale * weights, regularization=scale * R
    )
    np.testing.assert_allclose(operator.matmat(rhs), inverse @ rhs, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        operator.rmatmat(probe), inverse.T.conj() @ probe, rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize("method", ["normal_solve", "normal_pinv"])
def test_fit_owns_array_weights_before_caching_factors(method):
    rng = np.random.default_rng(1)
    A = rng.normal(size=(20, 7))
    rhs = rng.normal(size=20)
    weights = np.linspace(0.2, 1, 20)
    expected = np.linalg.lstsq(weights[:, None] * A, weights * rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A, sqrt_weights=weights)
    solver = LeastSquaresSolver(method)
    solver.solve(problem, rhs)
    weights[:] *= np.linspace(1, 20, 20)
    np.testing.assert_allclose(solver.solve(problem, rhs), expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("helmholtz", [False, True])
def test_transform_owns_weights_for_both_rhs_and_cached_factors(helmholtz):
    grid = GlobalCSBasis(4).native_grid
    basis = SHBasis(2, 2, mean_free=True)
    weights = np.linspace(0.2, 1, grid.size)
    transform = SphericalTransform(basis, grid, sqrt_weights=weights)
    rhs = np.random.default_rng(1).normal(size=(2, grid.size) if helmholtz else grid.size)
    analyze = transform.analyze_helmholtz if helmholtz else transform.analyze_scalar
    expected = analyze(rhs, solver="normal_solve")
    weights[: grid.size // 2] *= 10
    np.testing.assert_allclose(
        analyze(rhs, solver="normal_solve"), expected, rtol=1e-12, atol=1e-12
    )


def test_component_weights_do_not_silently_define_a_scalar_measure():
    basis = GlobalCSBasis(4)
    weights = np.vstack([np.ones(basis.coefficient_count), np.full(basis.coefficient_count, 2)])
    transform = SphericalTransform(basis, basis.native_grid, sqrt_weights=weights)
    # Scalar evaluation remains available; only the scalar fit lacks weights.
    np.testing.assert_array_equal(transform.synthesize_scalar(np.ones(basis.coefficient_count)), 1)
    with pytest.raises(ValueError, match="Component-specific sqrt_weights"):
        transform.analyze_scalar(np.ones(basis.coefficient_count))


@pytest.mark.parametrize("helmholtz", [False, True])
def test_regularized_native_cs_direct_fit_never_materializes_a_dense_operator(
    monkeypatch, helmholtz
):
    basis = GlobalCSBasis(4)
    transform = SphericalTransform(basis, basis.native_grid, area_weighted=True, reg_lambda=0.02)
    problem = (
        transform.helmholtz_least_squares_problem
        if helmholtz
        else transform.scalar_least_squares_problem
    )
    rhs = np.random.default_rng(9).normal(
        size=(2, basis.coefficient_count, 2) if helmholtz else (basis.coefficient_count, 2)
    )
    Z = (
        np.eye(problem.solution_size)
        if problem.constraints is None
        else null_space(problem.constraints)
    )
    system = problem.system_operator.to_sparse_matrix().toarray() @ Z
    rhs_block, batch_shape, _ = problem.assemble_rhs_block(rhs)
    reference = (Z @ np.linalg.lstsq(system, rhs_block, rcond=None)[0]).reshape(
        problem.solution_shape + batch_shape
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Sparse direct analysis must not create a dense matrix.")

    monkeypatch.setattr(LinearMap, "to_matrix", unexpected)
    monkeypatch.setattr(LinearMap, "_dense_array", unexpected)
    analyze = transform.analyze_helmholtz if helmholtz else transform.analyze_scalar
    actual = analyze(rhs, solver="normal_solve")
    np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-9)


def test_sparse_structure_survives_operator_algebra_without_affecting_action():
    from kompe.math import diagonal_linear_map, take_linear_map, vstack_linear_maps

    A = as_linear_map(csr_matrix(np.arange(20.0).reshape(5, 4)))
    weights = diagonal_linear_map(get_array_module().asarray(np.linspace(0.5, 1.5, 5)))
    select = take_linear_map((4,), [1, 3, 1])
    operator = vstack_linear_maps([2 * weights @ A, A]) @ select.adjoint()
    summed = operator.adjoint() @ operator + select @ select.adjoint()
    assert summed.is_sparse
    sparse = summed.to_sparse_matrix()
    assert summed.to_sparse_matrix() is sparse
    values = np.random.default_rng(3).normal(size=(3, 2))
    np.testing.assert_allclose(summed.matmat(values), sparse @ values)
    assert summed.materialized_matrix is None


def test_sparse_constraint_ownership_and_persistent_identity():
    C = csr_matrix([[1.0, 2.0, 1.0]])
    problem = LeastSquaresProblem(np.eye(3), constraints=C, cache_identity="same-data")
    C.data[0] = 3.0
    changed = LeastSquaresProblem(np.eye(3), constraints=C, cache_identity="same-data")
    np.testing.assert_array_equal(problem.constraints.data, [1.0, 2.0, 1.0])
    assert problem.reduced_problem.cache_identity != changed.reduced_problem.cache_identity


@pytest.mark.requires_jax
@pytest.mark.parametrize("jax_sparse", [False, True])
def test_sparse_constrained_direct_fit_can_initialize_inside_jit(jax_sparse):
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    A = csr_matrix(np.random.default_rng(4).normal(size=(9, 6)))
    problem = LeastSquaresProblem(
        BCOO.from_scipy_sparse(A) if jax_sparse else A,
        sqrt_weights=np.linspace(0.2, 1, 9),
        regularization=jnp.ones(6),
        constraints=np.ones((1, 6)),
    )
    rhs = jnp.arange(18.0).reshape(9, 2)
    solver = LeastSquaresSolver("normal_solve")
    with jax.checking_leaks():
        actual = jax.jit(lambda b: solver.solve(problem, b))(rhs)
    expected = LeastSquaresSolver("svd").solve(problem, rhs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.requires_jax
def test_jax_sparse_normal_diagonal_handles_duplicates_and_padding():
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    data = jnp.array([1.0, -1.0, 2.0, 9.0])
    indices = jnp.array([[0, 0], [0, 0], [1, 2], [2, 3]])
    sparse = BCOO((data, indices), shape=(2, 3))
    operator = as_linear_map(sparse)
    expected = np.asarray(sparse.todense())
    np.testing.assert_allclose(operator.to_sparse_matrix().toarray(), expected)
    np.testing.assert_allclose(operator.normal_matrix_diag(), np.sum(expected**2, axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_sparse_factors_preserve_precision_and_accept_a_wider_rhs(dtype):
    xp = get_array_module()
    data = 2 * np.eye(3, dtype=dtype)
    penalty = np.eye(3, dtype=dtype)
    operator = sparse_least_squares_map(data, regularization=penalty)
    rhs = xp.asarray([1.0, 2.0, 3.0], dtype=dtype)
    result = operator(rhs)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_allclose(result, 0.4 * rhs, rtol=5e-7)
    for factor in [1 + 2j, 0.5 - 1j]:
        wider = xp.asarray(rhs, dtype=np.complex128) * factor
        result = operator(wider)
        assert result.dtype == np.dtype(np.complex128)
        np.testing.assert_allclose(result, 0.4 * wider, rtol=1e-14)
        np.testing.assert_allclose(operator.adjoint()(wider), 0.4 * wider, rtol=1e-14)


@pytest.mark.parametrize("scale", [1e-20, 1e20])
def test_sparse_single_precision_scaling_does_not_square_extreme_units(scale):
    xp = get_array_module()
    data = np.eye(3, dtype=np.float32) * np.float32(scale)
    rhs = xp.asarray([1.0, 2.0, 3.0], dtype=np.float32) * np.float32(scale)
    operator = sparse_least_squares_map(data)
    result = operator(rhs)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-6)


@pytest.mark.requires_jax
def test_sparse_callback_honors_disabled_jax_x64():
    import jax
    import jax.numpy as jnp

    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        # Geometry may originate as CPU float64, but JAX controls its output dtype.
        operator = sparse_least_squares_map(np.eye(3, dtype=np.float64))
        result = jax.jit(operator.matvec)(jnp.ones(3, dtype=jnp.float32))
        assert result.dtype == jnp.float32
        np.testing.assert_array_equal(result, np.ones(3))
        assert not jax.config.x64_enabled
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("algorithm", ["lsmr", "cgls"])
def test_constrained_fit_accepts_a_full_space_initial_guess_and_preconditioner(algorithm):
    from kompe.math import identity_linear_map

    rng = np.random.default_rng(43)
    matrix = rng.normal(size=(12, 7))
    C = rng.normal(size=(2, 7))
    rhs = rng.normal(size=12)
    problem = LeastSquaresProblem(matrix, constraints=C)
    solver = LeastSquaresSolver(algorithm, tolerance=1e-12)
    preconditioner = identity_linear_map((7,))
    actual = solver.solve(problem, rhs, preconditioner=preconditioner, x0=np.arange(7.0))
    prepared = solver.prepare(problem, preconditioner=preconditioner)(rhs)
    Z = null_space(C)
    expected = Z @ np.linalg.lstsq(matrix @ Z, rhs, rcond=None)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(prepared, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("method", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_independent_nullspace_restriction_does_not_reimpose_roundoff(method, compact, scale):
    from kompe.math import null_space_linear_map

    rng = np.random.default_rng(43)
    A = rng.normal(size=(12, 7))
    C = rng.normal(size=(2, 7)) + 0.3j * rng.normal(size=(2, 7))
    rhs = rng.normal(size=(12, 2))
    Z = null_space(C)
    basis = scale * (null_space_linear_map(C) if compact else as_linear_map(Z))
    problem = LeastSquaresProblem(A, constraints=C)
    restricted = problem.restrict_solution(basis)
    assert restricted.constraints is None
    actual = basis(LeastSquaresSolver(method, tolerance=1e-12).solve(restricted, rhs))
    expected = Z @ np.linalg.lstsq(A @ Z, rhs, rcond=None)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("method", LeastSquaresSolver.VALID_SOLVERS)
def test_partial_coordinate_restriction_keeps_only_remaining_constraints(method):
    from kompe.math import take_linear_map

    p = LeastSquaresProblem(np.eye(5), constraints=np.eye(5)[:3])
    Z = take_linear_map((5,), [1, 2, 3, 4]).adjoint()
    restricted = p.restrict_solution(Z)
    assert restricted.constraints.shape == (2, 4)
    actual = Z(LeastSquaresSolver(method).solve(restricted, np.arange(1.0, 6.0)))
    np.testing.assert_allclose(actual, [0, 0, 0, 4, 5], atol=1e-13)


@pytest.mark.parametrize("method", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("sparse", [False, True])
def test_constraint_row_units_do_not_change_the_allowed_subspace(method, sparse):
    C = np.array([[1e-20, 0.0, 0.0, 0.0], [0.0, 0.0, 1e20, 0.0]])
    p = LeastSquaresProblem(csr_matrix(np.eye(4)) if sparse else np.eye(4), constraints=C)
    actual = LeastSquaresSolver(method).solve(p, np.arange(1.0, 5.0))
    np.testing.assert_allclose(actual, [0, 2, 0, 4], atol=1e-13)


@pytest.mark.parametrize("scale", [1e-20, 1e20])
@pytest.mark.parametrize("regularized", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_diagonal_direct_fit_preserves_units_without_squaring_them(scale, regularized, dtype):
    xp = get_array_module()
    phase = (1 + 0.5j) if np.issubdtype(dtype, np.complexfloating) else 1.0
    A = xp.asarray([phase, 2 * phase, 3 * phase], dtype=dtype)
    R = xp.asarray([0.2, 0.4, 0.6], dtype=dtype) if regularized else None
    b = xp.asarray([1.0, 2.0, 3.0], dtype=dtype)
    expected = (
        np.asarray(A).conj()
        * np.asarray(b)
        / (np.abs(np.asarray(A)) ** 2 + (0 if R is None else np.abs(np.asarray(R)) ** 2))
    )
    problem = LeastSquaresProblem(A * scale, regularization=None if R is None else R * scale)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = LeastSquaresSolver("normal_solve").solve(problem, b * scale)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_allclose(actual, expected, rtol=5e-7)


@pytest.mark.parametrize("method", ["lsmr", "cgls"])
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("batched_initial", [False, True])
def test_warm_starts_use_original_shaped_coefficients(method, constrained, batched_initial):
    # Two observable directions and one unconstrained, unobservable direction.
    # A right preconditioner must not rescale the initial nullspace component.
    A = np.eye(4)[:2]
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]])
    initial = np.array([0.0, 0.0, 3.0, 0.0]).reshape(2, 2)
    if batched_initial:
        initial = np.stack([initial, 2 * initial], axis=-1)
    p = LeastSquaresProblem(
        as_linear_map(A, input_shape=(2, 2)),
        constraints=np.array([[0.0, 0.0, 0.0, 1.0]]) if constrained else None,
    )
    P = as_linear_map(np.array([2.0, 4.0, 6.0, 8.0]), input_shape=(2, 2), output_shape=(2, 2))
    actual = LeastSquaresSolver(method, tolerance=1e-12).solve(
        p, rhs, preconditioner=P, x0=initial
    )
    expected = np.vstack([rhs, [3.0, 6.0 if batched_initial else 3.0], [0.0, 0.0]]).reshape(
        2, 2, 2
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["lsmr", "cgls"])
def test_warm_start_does_not_shift_an_explicit_regularization_penalty(method):
    rng = np.random.default_rng(32)
    A = rng.normal(size=(10, 6))
    R = rng.normal(size=(4, 6))
    C = rng.normal(size=(1, 6))
    rhs = rng.normal(size=(10, 3))
    initial = rng.normal(size=(2, 3, 3))
    problem = LeastSquaresProblem(
        as_linear_map(A, input_shape=(2, 3)),
        regularization=R,
        constraints=C,
    )
    solver = LeastSquaresSolver(method, tolerance=1e-12, preconditioner="jacobi")
    actual = solver.solve(problem, rhs, x0=initial)
    Z = null_space(C)
    expected = (
        Z
        @ np.linalg.lstsq(
            np.vstack([A, R]) @ Z,
            np.vstack([rhs, np.zeros((4, 3))]),
            rcond=None,
        )[0]
    )
    np.testing.assert_allclose(actual, expected.reshape(2, 3, 3), rtol=1e-10, atol=1e-11)


def test_constraint_rank_is_independent_of_unrelated_coordinate_column_units():
    p = LeastSquaresProblem(np.eye(3), constraints=np.array([[1.0, 0.0, 0.0]]))
    Z = as_linear_map(np.diag([1e-20, 1e20, 1.0]))
    restricted = p.restrict_solution(Z)
    assert restricted.constraints.shape == (1, 3)
    np.testing.assert_allclose(restricted.solution_basis.to_matrix()[0], 0.0, atol=1e-13)


def test_single_precision_restriction_does_not_drop_a_resolved_constraint():
    from kompe.math import identity_linear_map

    n = 512
    C = np.eye(1, n, dtype=np.float32)
    Z = np.eye(n, dtype=np.float32)[:, 1:]
    Z[0, 0] = 1e-4
    problem = LeastSquaresProblem(identity_linear_map(n, dtype=np.float32), constraints=C)
    restricted = problem.restrict_solution(Z)
    assert restricted.constraints.shape == (1, n - 1)
    np.testing.assert_allclose(restricted.constraints[0, 1:], 0.0, atol=1e-12)
    assert abs(restricted.constraints[0, 0]) > 0.9
