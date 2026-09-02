"""Tensor pseudoinverses, weights, and backend preservation."""

import numpy as np
import pytest

from kompe.math import backend_context, get_array_module, pseudoinverse, weighted_tensor_pinv


def test_weighted_tensor_pinv_matches_explicit_weighted_least_squares():
    """Weighted pseudoinverse solves weighted normal equations."""
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    sqrt_weights = np.array([1.0, 1.5, 2.0, 2.5])
    weight_matrix = np.diag(sqrt_weights**2)

    actual = weighted_tensor_pinv(A, sqrt_weights=sqrt_weights, n_leading_flattened=1)
    expected = np.linalg.solve(A.T @ weight_matrix @ A, A.T @ weight_matrix)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
@pytest.mark.parametrize("shape, leading", [((6, 4), 1), ((2, 3, 4), 2), ((2, 3, 2, 2), 2)])
@pytest.mark.parametrize("complex_values", [False, True])
def test_weighted_tensor_pinv_reuses_tensor_analysis(
    monkeypatch, backend, shape, leading, complex_values
):
    """Weighting preserves axes and uses one canonical pseudoinverse."""
    rng = np.random.default_rng(123)
    values = rng.normal(size=shape)
    if complex_values:
        values = values + 1j * rng.normal(size=shape)
    data_shape, coefficient_shape = shape[:leading], shape[leading:]
    weights = np.linspace(0.0, 2.0, int(np.prod(data_shape)))
    rtol = 1e-12
    matrix = values.reshape(weights.size, -1)
    expected = np.linalg.pinv(weights[:, None] * matrix, rtol=rtol) * weights
    calls = []
    tensor_pinv = pseudoinverse.tensor_pinv

    def record_tensor_pinv(A, n_leading_flattened=2, rtol=1e-15):
        calls.append((A.shape, n_leading_flattened, rtol))
        return tensor_pinv(A, n_leading_flattened=n_leading_flattened, rtol=rtol)

    monkeypatch.setattr(pseudoinverse, "tensor_pinv", record_tensor_pinv)
    with backend_context(backend):
        xp = get_array_module()
        actual = weighted_tensor_pinv(
            xp.asarray(values), xp.asarray(weights), n_leading_flattened=leading, rtol=rtol
        )
        assert isinstance(actual, xp.ndarray)

    assert calls == [(shape, leading, rtol)]
    assert actual.shape == coefficient_shape + data_shape
    np.testing.assert_allclose(actual, expected.reshape(actual.shape), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_weighted_tensor_pinv_without_weights_preserves_cutoff(backend):
    """Omitting weights preserves the ordinary singular-value cutoff."""
    with backend_context(backend):
        xp = get_array_module()
        values = xp.asarray([[2.0, 0.0], [0.0, 1e-8]])
        actual = weighted_tensor_pinv(values, n_leading_flattened=1, rtol=1e-6)
        assert isinstance(actual, xp.ndarray)
    np.testing.assert_allclose(actual, np.diag([0.5, 0.0]), atol=1e-12)


@pytest.mark.requires_jax
def test_explicit_jax_weights_preserve_jax_analysis(monkeypatch):
    """JAX weights choose JAX even with a NumPy matrix and default."""
    import jax.numpy as jnp

    def fail_numpy_pinv(*args, **kwargs):
        raise AssertionError("JAX-weighted analysis must not factorize on NumPy")

    with backend_context("numpy"):
        monkeypatch.setattr(np.linalg, "pinv", fail_numpy_pinv)
        result = weighted_tensor_pinv(np.eye(2), jnp.asarray([1.0, 0.0]), n_leading_flattened=1)
        assert isinstance(result, jnp.ndarray)
    np.testing.assert_allclose(result, np.diag([1.0, 0.0]), atol=1e-12)
