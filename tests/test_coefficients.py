"""Coefficient-array shape, gauge, and basis semantics."""

import numpy as np
import pytest

from kompe import GlobalCSBasis, SECSBasis, SHBasis, SphericalGrid, SphericalTransform
from kompe.basis import BasisSubset
from kompe.coefficients import CoefficientSpace


def test_coefficient_space_applies_scalar_mean_free_projection():
    """The space applies its explicit scalar mean-free policy."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="scalar", mean_free=True)
    coeffs = np.linspace(0.0, 1.0, basis.coefficient_count) + 2.0

    field = field_space.project_mean_free(coeffs)

    assert field_space.basis is basis
    assert field_space.mean_free
    np.testing.assert_allclose(basis.scalar_mean(field), 0.0, atol=1e-12)
    assert field.shape == coeffs.shape
    assert field.shape == (basis.coefficient_count,)
    np.testing.assert_allclose(field.reshape(-1), field.reshape(-1))


def test_coefficient_space_preserves_tangential_shape():
    """Tangential coefficient fields keep their two-component layout."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="helmholtz", mean_free=True)
    coeffs = np.vstack(
        [
            np.linspace(0.0, 1.0, basis.coefficient_count) + 1.0,
            np.linspace(1.0, 2.0, basis.coefficient_count) - 0.5,
        ]
    )

    field = field_space.project_mean_free(coeffs)

    assert field_space.representation == "helmholtz"
    assert field.shape == (2, basis.coefficient_count)
    np.testing.assert_allclose(field.reshape(-1), field.reshape(-1))
    np.testing.assert_allclose(basis.scalar_mean(field.T), np.zeros(2), atol=1e-12)


def test_coefficient_space_canonicalizes_flat_tangential_coefficients():
    """Flat tangential input is stored as component x coefficient."""
    basis = SHBasis(3, 2)
    field_space = CoefficientSpace(basis, representation="helmholtz")
    coeffs = np.arange(2 * basis.coefficient_count)

    field = field_space.project_mean_free(coeffs)

    assert field_space.shape == (2, basis.coefficient_count)
    assert field.shape == field_space.shape
    np.testing.assert_array_equal(field, coeffs.reshape(2, basis.coefficient_count))
    np.testing.assert_array_equal(field.reshape(-1), coeffs)


def test_field_space_infers_an_intrinsically_mean_free_basis():
    """Infer when the basis itself omits the mean mode."""
    field_space = CoefficientSpace(SHBasis(3, 2, mean_free=True))

    assert field_space.mean_free


@pytest.mark.parametrize("representation", ["scalar", "helmholtz"])
@pytest.mark.parametrize("flat", [False, True])
def test_coefficient_space_normalizes_batches_without_losing_axes(representation, flat):
    """Trailing batch axes survive normalization in nodal coefficient space."""
    basis = GlobalCSBasis(4)
    space = CoefficientSpace(basis, representation=representation, mean_free=True)
    values = np.random.default_rng(391).normal(size=space.shape + (3, 4))
    inputs = values.reshape(space.size, 3, 4) if flat else values
    actual = space.project_mean_free(inputs)
    coefficient_axis = len(space.shape) - 1
    means = np.tensordot(basis.scalar_mean_weights, values, axes=([0], [coefficient_axis]))
    expected = values - np.expand_dims(means, coefficient_axis)
    assert actual.shape == values.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        basis.scalar_mean(np.moveaxis(actual, coefficient_axis, 0)), 0.0, atol=1e-14
    )


def test_cs_subset_without_a_constant_is_not_intrinsically_mean_free():
    """Omitting a node removes the constant, not every nonzero-mean field."""
    parent = GlobalCSBasis(4)
    subset = BasisSubset(parent, np.arange(parent.coefficient_count - 1))
    space = CoefficientSpace(subset)
    field = space.project_mean_free(np.ones(subset.coefficient_count))

    assert subset.omits_constant_mode()
    assert not subset.mean_free
    assert not space.mean_free
    assert subset.scalar_mean(field) > 0.9
    np.testing.assert_array_equal(field, 1.0)
    with pytest.raises(ValueError, match="constant shift"):
        CoefficientSpace(subset, mean_free=True).project_mean_free(field)
    with pytest.raises(ValueError, match="constant shift"):
        subset.project_helmholtz_mean_free(np.ones((2, subset.coefficient_count)))


def test_sh_subset_derives_mean_free_status_without_override_metadata():
    """Removing the monopole makes any further subset intrinsically mean-free."""
    full = SHBasis(3, 2, mean_free=False)
    subset = BasisSubset(full, np.flatnonzero(full.n > 0))
    for basis in (subset, BasisSubset(subset, np.arange(3))):
        assert basis.mean_free
        assert basis.omits_constant_mode()
        assert basis.with_mean_free(True) is basis
        assert CoefficientSpace(basis).mean_free
        values = np.ones(basis.coefficient_count)
        assert basis.project_scalar_mean_free(values) is values
        np.testing.assert_allclose(basis.scalar_mean(values), 0.0)


@pytest.mark.parametrize("kind", ["SH", "CS", "permuted_CS"])
@pytest.mark.parametrize("flat", [False, True])
def test_shared_mean_projection_removes_only_constant_potentials(kind, flat):
    """Batched potentials retain their tangential fields without layout conversion."""
    cs = GlobalCSBasis(4)
    basis = SHBasis(3, 2, mean_free=False) if kind == "SH" else cs
    if kind == "permuted_CS":
        basis = BasisSubset(cs, np.arange(cs.coefficient_count)[::-1])
        assert not basis.scalar_constant_coefficients.flags.writeable
    assert not basis.mean_free
    assert not basis.omits_constant_mode()
    n = basis.coefficient_count
    values = np.random.default_rng(781).normal(size=(2, n, 3))
    inputs = values.reshape(2 * n, 3) if flat else values
    actual = np.asarray(basis.project_helmholtz_mean_free(inputs)).reshape(values.shape)
    means = np.einsum("cnt,n->ct", values, basis.scalar_mean_weights)
    expected = values - means[:, None, :] * basis.scalar_constant_coefficients[None, :, None]
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(basis.scalar_mean(np.moveaxis(actual, 1, 0)), 0.0, atol=1e-13)

    transform = SphericalTransform(basis, cs.native_grid)
    np.testing.assert_allclose(
        transform.synthesize_helmholtz(actual),
        transform.synthesize_helmholtz(values),
        rtol=1e-12,
        atol=1e-12,
    )


def test_coefficient_space_validates_coefficient_length():
    """Coefficient arrays must match the declared basis."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = CoefficientSpace(basis, representation="scalar")

    with pytest.raises(ValueError, match="coefficients"):
        field_space.validate_coefficients(np.zeros(basis.coefficient_count + 1))


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


def test_field_space_applies_mean_constraint_after_sample_analysis():
    """Field-space constraints compose with sample analysis."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="scalar", mean_free=True)
    grid = basis.mesh.cell_centers
    values = np.linspace(0.0, 1.0, basis.coefficient_count) + 3.0
    transform = SphericalTransform(basis, grid)
    analyzed = transform.analyze_scalar_samples(values, input_grid=grid)
    constrained = field_space.project_mean_free(analyzed)
    assert analyzed.shape == (basis.coefficient_count,)
    np.testing.assert_allclose(basis.scalar_mean(constrained), 0.0, atol=1e-12)
