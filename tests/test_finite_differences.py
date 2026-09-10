"""Polynomial-exact finite-difference stencil construction."""

from math import factorial

import numpy as np
import pytest

from kompe.math import centered_derivative, finite_difference_weights, get_array_module


@pytest.mark.parametrize("points", [[-1, 0, 1], [0, 1, 2, 3], [-3, -1, 0, 2, 4]])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("spacing", [0.25, 2.0])
def test_stencil_differentiates_polynomials_at_the_origin(points, order, spacing):
    """Centered, one-sided, and uneven stencils preserve derivative units."""
    weights = finite_difference_weights(points, order=order, h=spacing)
    positions = spacing * np.asarray(points)
    for power in range(len(points)):
        expected = factorial(order) if power == order else 0.0
        assert weights @ positions**power == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("half_window_points", [1, 2])
@pytest.mark.parametrize("n_points", [0, 1, 3, 8])
def test_centered_derivative_preserves_backend_windows_and_missing_data(
    half_window_points, n_points
):
    """Batched derivatives retain spacing and missing stencil values."""
    xp = get_array_module()
    x = np.array([0, 1, 4, 5, 8, 10, 15, 17], dtype=float)[:n_points]
    values = np.stack([2 * x, -3 * x]).reshape(1, 2, n_points)
    if n_points > 1:
        values[0, 1, 1] = np.nan
    expected = np.full_like(values, np.nan)
    for i in range(half_window_points, n_points - half_window_points):
        left, right = i - half_window_points, i + half_window_points
        stencil = values[..., [left, i, right]]
        expected[..., i] = np.where(np.isfinite(stencil).all(axis=-1), [[2.0, -3.0]], np.nan)
    result = centered_derivative(
        xp.asarray(x), xp.asarray(values), half_window_points=half_window_points
    )
    assert isinstance(result, xp.ndarray)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("half_window_points", [1, 2])
@pytest.mark.parametrize("complex_values", [False, True])
def test_centered_derivative_is_quadratic_exact_on_irregular_coordinates(
    half_window_points, complex_values
):
    """Differentiate at the sample, not its neighbours' midpoint."""
    xp = get_array_module()
    x = xp.asarray([0.0, 0.1, 0.3, 0.4, 1.0, 1.2, 2.0])
    scale = 1 + 2j if complex_values else 1
    values = scale * xp.stack([x**2, 3 * x**2 - 2 * x + 5])
    h = half_window_points

    actual = centered_derivative(x, values, half_window_points=h)
    expected = scale * xp.stack([2 * x, 6 * x - 2])

    np.testing.assert_allclose(actual[..., h:-h], expected[..., h:-h], atol=1e-12, rtol=1e-12)
    assert isinstance(actual, xp.ndarray)


def test_centered_derivative_validates_coordinates_and_stencil():
    """Bad numerical boundary inputs fail without repaired coordinates."""
    for x in ([0, 2, 1], [0, 1, 1], [0, np.nan, 2]):
        with pytest.raises(ValueError, match="strictly increasing"):
            centered_derivative(x, np.ones(3))
    with pytest.raises(ValueError, match="must match"):
        centered_derivative([0, 1, 2], np.ones((3, 2)))
    for window in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            centered_derivative([0, 1, 2], np.ones(3), half_window_points=window)


@pytest.mark.requires_jax
def test_centered_derivative_supports_jax_tracing_and_differentiation():
    """The stencil remains a pure array calculation under jit and autodiff."""
    import jax
    import jax.numpy as jnp

    x = jnp.asarray([0.0, 0.1, 0.3, 0.4, 1.0])
    result = jax.jit(centered_derivative)(x, x**2)
    np.testing.assert_allclose(result[1:-1], 2 * x[1:-1], atol=1e-12)
    slope = jax.grad(lambda scale: jnp.sum(centered_derivative(x, scale * x**2)[1:-1]))(1.0)
    np.testing.assert_allclose(slope, 2 * np.sum(x[1:-1]), atol=1e-12)
