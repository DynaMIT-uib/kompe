"""Affine propagation, including forcing that has no equilibrium."""

import numpy as np
import pytest

from kompe.math import affine_exponential, as_linear_map, diagonal_linear_map, get_array_module


def test_affine_exponential_integrates_a_forced_jordan_block():
    """x' = y, y' = 1 has a quadratic solution and no equilibrium."""
    xp = get_array_module()
    A = as_linear_map(xp.array([[0.0, 1.0], [0.0, 0.0]]))
    P, q = affine_exponential(A, xp.array([0.0, 1.0]), 0.3)
    np.testing.assert_allclose(P(xp.array([2.0, 3.0])) + q, [2.945, 3.3], atol=1e-14)


def test_affine_exponential_preserves_diagonal_and_scientific_axes(monkeypatch):
    xp = get_array_module()
    rates = xp.array([0.0, -1.0, -2.0, 0.5])
    A = diagonal_linear_map(rates, input_shape=(2, 2), output_shape=(2, 2))
    monkeypatch.setattr(
        type(A), "to_matrix", lambda self, **kwargs: pytest.fail("Diagonal densified.")
    )
    b = xp.ones((2, 2))
    P, q = affine_exponential(A, b, 0.2)
    assert P.is_diagonal
    assert q.shape == (2, 2)
    expected_q = np.array([0.2, -np.expm1(-0.2), -np.expm1(-0.4) / 2, 2 * np.expm1(0.1)])
    np.testing.assert_allclose(q.reshape(-1), expected_q, atol=1e-15)
    np.testing.assert_allclose(P(b).reshape(-1), np.exp(0.2 * np.asarray(rates)), atol=1e-14)


@pytest.mark.parametrize("complex_values", [False, True])
def test_affine_exponential_composes_and_matches_independent_ode(complex_values):
    from scipy.integrate import solve_ivp

    xp = get_array_module()
    A = np.array([[-2.0, 3.0], [0.5, -1.0]])
    if complex_values:
        A = A + 1j * np.diag([0.2, -0.4])
    b = np.array([0.7, -0.2])
    x = np.array([0.4, 0.8], dtype=A.dtype)
    P, q = affine_exponential(xp.asarray(A), xp.asarray(b), 0.2)
    P2, q2 = affine_exponential(xp.asarray(A), xp.asarray(b), 0.4)
    actual = P(P(xp.asarray(x)) + q) + q
    reference = solve_ivp(lambda t, values: A @ values + b, (0, 0.4), x, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual, reference.y[:, -1], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(actual, P2(xp.asarray(x)) + q2, rtol=1e-13, atol=1e-14)


def test_zero_duration_is_identity():
    P, q = affine_exponential([[0.0, 1.0], [-2.0, -3.0]], [2.0, 4.0], 0)
    np.testing.assert_allclose(P(np.array([3.0, 5.0])), [3.0, 5.0])
    np.testing.assert_array_equal(q, 0.0)


@pytest.mark.parametrize("scale", [1e-20, 1e20])
def test_forcing_units_do_not_change_propagation(scale):
    xp = get_array_module()
    A = xp.array([[-1.0, 2.0], [0.0, -2.0]])
    b = xp.array([1.0, 2.0])
    P, q = affine_exponential(A, b, 0.2)
    scaled_P, scaled_q = affine_exponential(A, scale * b, 0.2)
    np.testing.assert_allclose(scaled_q / scale, q, rtol=1e-13)
    np.testing.assert_allclose(scaled_P(b), P(b), rtol=1e-13)


@pytest.mark.parametrize("duration", [-1, np.inf, np.nan, True])
def test_affine_exponential_validates_duration(duration):
    with pytest.raises((TypeError, ValueError), match="duration"):
        affine_exponential(np.eye(2), np.ones(2), duration)


def test_affine_exponential_validates_shapes():
    with pytest.raises(ValueError, match="shape"):
        affine_exponential(np.ones((2, 3)), np.ones(3), 1.0)
    with pytest.raises(ValueError, match="shape"):
        affine_exponential(np.eye(2), np.ones((2, 1)), 1.0)


@pytest.mark.parametrize("materialized", [False, True])
def test_matrix_free_and_materialized_generators_agree(materialized):
    from kompe.math import LinearMap

    xp = get_array_module()
    matrix = xp.array([[-2.0, 1.0], [0.0, -3.0]])
    A = LinearMap(
        shape=(2, 2),
        dtype=matrix.dtype,
        matvec=lambda values: matrix @ values,
        rmatvec=lambda values: matrix.T @ values,
    )
    if materialized:
        A.to_matrix()
    P, q = affine_exponential(A, xp.ones(2), 0.3)
    expected_P, expected_q = affine_exponential(matrix, xp.ones(2), 0.3)
    values = xp.array([0.2, -0.1])
    np.testing.assert_allclose(P(values) + q, expected_P(values) + expected_q, atol=1e-14)
    if xp is not np:
        from jax import jit

        np.testing.assert_allclose(jit(lambda x: P(x) + q)(values), P(values) + q, atol=1e-14)
