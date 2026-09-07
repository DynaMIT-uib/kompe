"""Tests for field coefficient values."""

import numpy as np
import pytest

from kompe import GlobalCSBasis, SECSBasis, SHBasis, SphericalGrid
from kompe.coefficients import CoefficientSpace, FieldCoefficients
from kompe.math import backend_context


def test_field_coefficients_numpy_copy_contract():
    """NumPy coercion distinguishes views, copies, and dtype changes."""
    with backend_context("numpy"):
        basis = SHBasis(2, 1)
        field = FieldCoefficients(CoefficientSpace(basis), np.ones(basis.coefficient_count))
    view = np.array(field, copy=False)
    assert np.shares_memory(view, field.array)
    assert not view.flags.writeable
    copy = np.array(field, copy=True)
    assert not np.shares_memory(copy, field.array)
    copy[:] = 2.0
    np.testing.assert_array_equal(field.array, 1.0)
    assert np.asarray(field, dtype=np.float32).dtype == np.float32


def test_field_coefficients_applies_scalar_mean_free_projection():
    """FieldCoefficients applies scalar mean-free semantics."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="scalar", mean_free=True)
    coeffs = np.linspace(0.0, 1.0, basis.coefficient_count) + 2.0

    field = FieldCoefficients(field_space, coeffs)

    assert field.field_space is field_space
    assert field.field_space.basis is basis
    assert field.field_space.mean_free
    np.testing.assert_allclose(basis.scalar_mean(field.array), 0.0, atol=1e-12)
    assert field.array.shape == coeffs.shape
    assert field.array.shape == (basis.coefficient_count,)
    np.testing.assert_allclose(field.to_vector(), field.array.reshape(-1))


def test_field_coefficients_preserves_tangential_shape():
    """Tangential coefficient fields keep their two-component layout."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="helmholtz", mean_free=True)
    coeffs = np.vstack(
        [
            np.linspace(0.0, 1.0, basis.coefficient_count) + 1.0,
            np.linspace(1.0, 2.0, basis.coefficient_count) - 0.5,
        ]
    )

    field = FieldCoefficients(field_space, coeffs)

    assert field.field_space.representation == "helmholtz"
    assert field.array.shape == (2, basis.coefficient_count)
    np.testing.assert_allclose(field.to_vector(), field.array.reshape(-1))
    np.testing.assert_allclose(basis.scalar_mean(field.array), np.zeros(2), atol=1e-12)


def test_field_coefficients_canonicalizes_flat_tangential_coefficients():
    """Flat tangential input is stored as component x coefficient."""
    basis = SHBasis(3, 2)
    field_space = CoefficientSpace(basis, representation="helmholtz")
    coeffs = np.arange(2 * basis.coefficient_count)

    field = FieldCoefficients(field_space, coeffs)

    assert field.field_space.shape == (2, basis.coefficient_count)
    assert field.array.shape == field.field_space.shape
    np.testing.assert_array_equal(field.array, coeffs.reshape(2, basis.coefficient_count))
    np.testing.assert_array_equal(field.to_vector(), coeffs)


def test_field_coefficients_owns_immutable_numpy_values():
    """External mutation cannot invalidate cached operators."""
    basis = SHBasis(3, 2)
    field_space = CoefficientSpace(basis)
    source = np.arange(basis.coefficient_count, dtype=float)
    field = FieldCoefficients(field_space, source)
    source[:] = -1.0

    np.testing.assert_array_equal(field.array, np.arange(basis.coefficient_count, dtype=float))
    with pytest.raises((TypeError, ValueError)):
        field.array[0] = 10.0


def test_field_space_infers_an_intrinsically_mean_free_basis():
    """Infer when the basis itself omits the mean mode."""
    field_space = CoefficientSpace(SHBasis(3, 2, mean_free=True))

    assert field_space.mean_free


def test_field_coefficients_validates_coefficient_length():
    """FieldCoefficients rejects wrong coefficient lengths."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = CoefficientSpace(basis, representation="scalar")

    with pytest.raises(ValueError, match="FieldCoefficients.array"):
        FieldCoefficients(field_space, np.zeros(basis.coefficient_count + 1))


def test_field_space_rejects_sample_grid_as_coefficient_basis():
    """Sample locations are not coefficient representations."""
    grid = SphericalGrid(theta=[30.0, 60.0], phi=[0.0, 90.0])

    with pytest.raises(TypeError, match="ScalarBasis"):
        CoefficientSpace(grid)


def test_mean_free_field_space_requires_a_surface_basis():
    """Green-function coefficients have no surface-gauge semantics."""
    poles = SphericalGrid(theta=[30.0, 60.0], phi=[0.0, 90.0])
    basis = SECSBasis(poles, current_type="curl_free")

    with pytest.raises(TypeError, match="SurfaceDifferentialBasis"):
        CoefficientSpace(basis, mean_free=True)
