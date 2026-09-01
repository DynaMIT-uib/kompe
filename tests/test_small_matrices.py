"""Small-matrix formulas independent of spherical geometry."""

import numpy as np
import pytest

from kompe.math import backend_context, determinant_3x3, get_array_module, inverse_3x3


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_batched_small_matrix_formulas_match_dense_linalg(backend):
    """General nonsymmetric matrices retain the selected array backend."""
    matrices = np.random.default_rng(4).normal(size=(5, 3, 3)) + 4.0 * np.eye(3)
    with backend_context(backend):
        xp = get_array_module()
        determinants = determinant_3x3(xp.asarray(matrices))
        inverses = inverse_3x3(xp.asarray(matrices))
        assert isinstance(determinants, xp.ndarray)
        assert isinstance(inverses, xp.ndarray)
    np.testing.assert_allclose(determinants, np.linalg.det(matrices), rtol=1e-13)
    np.testing.assert_allclose(inverses, np.linalg.inv(matrices), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("operation", [determinant_3x3, inverse_3x3])
def test_small_matrix_formulas_require_a_batch_of_3x3_matrices(operation):
    """The specialized shape contract is explicit."""
    with pytest.raises(ValueError, match="shape"):
        operation(np.eye(2)[None, :, :])
