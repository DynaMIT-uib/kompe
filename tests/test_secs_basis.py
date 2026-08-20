"""Tests for SECS as a first-class spherical representation."""

import numpy as np
import pytest

from kompe import (
    RegionalCSMesh,
    RegionalCSProjection,
    ScalarBasis,
    SECSBasis,
    SphericalGrid,
    SurfaceDifferentialBasis,
)
from kompe.math import backend_context
from kompe.secs import (
    angular_distance,
    current_wedge_magnetic_field_matrices,
    magnetic_field_matrices,
    scalar_green_matrix,
    surface_current_matrices,
)


@pytest.fixture
def secs_basis():
    return SECSBasis(
        SphericalGrid(lat=[65.0, 70.0, 73.0], lon=[5.0, 18.0, 32.0]),
        current_type="divergence_free",
    )


@pytest.fixture
def evaluation_grid():
    return SphericalGrid(lat=[61.0, 67.0, 72.0, 76.0], lon=[-2.0, 12.0, 24.0, 40.0])


def test_angular_distance_preserves_coincident_and_antipodal_geometry():
    """Roundoff clipping must not move points away from 0 or 180 degrees."""
    distances = angular_distance(
        lat=np.array([30.0, -30.0]),
        lon=np.array([40.0, -140.0]),
        pole_latitudes=np.array([30.0, -30.0]),
        pole_longitudes=np.array([40.0, -140.0]),
        return_degrees=True,
    )

    np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-6)
    np.testing.assert_allclose(distances[0, 1], 180.0, atol=1e-6)


def test_secs_is_scalar_synthesis_without_closed_surface_claims(secs_basis):
    assert isinstance(secs_basis, ScalarBasis)
    assert not isinstance(secs_basis, SurfaceDifferentialBasis)
    assert isinstance(secs_basis, SECSBasis)
    assert secs_basis.kind == "SECS"
    assert secs_basis.index_length == 3
    assert secs_basis.index_names == ("latitude", "longitude")
    with pytest.raises(NotImplementedError, match="surface-current synthesis"):
        secs_basis.scalar_evaluation_matrix(secs_basis.poles, derivative="theta")


def test_secs_accepts_regional_grid_for_poles_and_evaluation():
    regional = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 25.0),
        600.0,
        500.0,
        shape=(4, 3),
        radius=6371.2,
    )
    basis = SECSBasis(poles=regional.cell_centers, current_type="curl_free")
    matrix = basis.scalar_evaluation_matrix(regional)

    assert basis.index_length == regional.size
    assert matrix.shape == (regional.size, regional.size)


def test_secs_scalar_synthesis_has_explicit_physical_mode(evaluation_grid):
    poles = SphericalGrid(lat=[65.0, 70.0], lon=[5.0, 18.0])
    curl_free = SECSBasis(poles=poles, current_type="curl_free")
    divergence_free = SECSBasis(poles=poles, current_type="divergence_free")

    expected_potential = scalar_green_matrix(
        evaluation_grid.lat,
        evaluation_grid.lon,
        poles.lat,
        poles.lon,
        quantity="potential",
        normalization=curl_free.normalization,
    )
    expected_current_magnitude = scalar_green_matrix(
        evaluation_grid.lat,
        evaluation_grid.lon,
        poles.lat,
        poles.lon,
        quantity="current_magnitude",
        normalization=divergence_free.normalization,
    )

    np.testing.assert_allclose(
        curl_free.scalar_evaluation_matrix(evaluation_grid), expected_potential
    )
    np.testing.assert_allclose(
        divergence_free.scalar_evaluation_matrix(evaluation_grid), expected_current_magnitude
    )
    assert curl_free.signature != divergence_free.signature


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
def test_surface_current_kernel_matches_canonical_components(
    secs_basis, evaluation_grid, current_type
):
    basis = SECSBasis(secs_basis.poles, current_type=current_type)
    east, north = surface_current_matrices(
        evaluation_grid.lat,
        evaluation_grid.lon,
        secs_basis.poles.lat,
        secs_basis.poles.lon,
        current_type=current_type,
        normalization=basis.normalization,
        source_radius=basis.radius,
    )
    canonical = basis.surface_current_matrix(evaluation_grid)

    np.testing.assert_allclose(canonical[0], -north)
    np.testing.assert_allclose(canonical[1], east)
    operator = basis.surface_current_operator(evaluation_grid)
    coefficients = np.array([0.4, -1.2, 0.7])
    np.testing.assert_allclose(
        operator @ coefficients,
        np.tensordot(canonical, coefficients, axes=1).reshape(-1),
    )


def test_two_component_secs_helmholtz_operator_matches_tensor(secs_basis, evaluation_grid):
    matrix = secs_basis.helmholtz_current_synthesis_matrix(evaluation_grid)
    operator = secs_basis.helmholtz_current_synthesis_operator(evaluation_grid)
    coefficients = np.array([[0.2, -0.5, 0.8], [1.0, 0.3, -0.4]])

    assert matrix.shape == (2, evaluation_grid.size, 2, secs_basis.index_length)
    np.testing.assert_allclose(
        operator @ coefficients.reshape(-1),
        np.tensordot(matrix, coefficients, axes=2).reshape(-1),
    )


def test_chunked_secs_current_operator_matches_dense_forward_and_adjoint(
    secs_basis, evaluation_grid
):
    dense = secs_basis.surface_current_operator(evaluation_grid)
    chunked = secs_basis.surface_current_operator(evaluation_grid, chunk_size=2)
    coefficients = np.array([0.4, -1.2, 0.7])
    values = np.linspace(-1.0, 1.0, 2 * evaluation_grid.size)
    coefficient_block = np.column_stack([coefficients, -2.0 * coefficients])
    value_block = np.column_stack([values, 0.5 * values])

    np.testing.assert_allclose(chunked @ coefficients, dense @ coefficients)
    np.testing.assert_allclose(chunked.rmatvec(values), dense.rmatvec(values))
    np.testing.assert_allclose(chunked.matmat(coefficient_block), dense.matmat(coefficient_block))
    np.testing.assert_allclose(chunked.rmatmat(value_block), dense.rmatmat(value_block))


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
def test_magnetic_field_uses_canonical_radial_theta_phi_order(
    secs_basis, evaluation_grid, current_type
):
    basis = SECSBasis(secs_basis.poles, current_type=current_type)
    evaluation_radius = 6371.2e3
    east, north, radial = magnetic_field_matrices(
        evaluation_grid.lat,
        evaluation_grid.lon,
        evaluation_radius,
        secs_basis.poles.lat,
        secs_basis.poles.lon,
        current_type=current_type,
        normalization=basis.normalization,
        source_radius=basis.radius,
    )
    canonical = basis.magnetic_field_matrix(evaluation_grid, evaluation_radius)
    np.testing.assert_allclose(canonical, np.stack([radial, -north, east]))


def test_secs_kernel_does_not_mutate_numpy_error_policy(secs_basis, evaluation_grid):
    before = np.geterr().copy()
    secs_basis.magnetic_field_matrix(evaluation_grid, 6371.2e3)
    assert np.geterr() == before


def test_induction_image_current_uses_requested_normalization(secs_basis, evaluation_grid):
    arguments = (
        evaluation_grid.lat,
        evaluation_grid.lon,
        6371.2e3,
        secs_basis.poles.lat,
        secs_basis.poles.lon,
    )
    options = {
        "current_type": "divergence_free",
        "source_radius": secs_basis.radius,
        "induction_nullification_radius": 6371.2e3,
    }
    reference = magnetic_field_matrices(
        *arguments, normalization=secs_basis.normalization, **options
    )
    scaled = magnetic_field_matrices(
        *arguments, normalization=3 * secs_basis.normalization, **options
    )

    # At the nullification radius, primary and image fields cancel to roundoff.
    # The absolute tolerance remains fourteen orders below either constituent field.
    for actual, expected in zip(scaled, reference, strict=True):
        np.testing.assert_allclose(actual, 3 * expected, rtol=1e-12, atol=1e-25)


def test_secs_rejects_invalid_current_type(secs_basis, evaluation_grid):
    with pytest.raises(ValueError, match="current_type"):
        SECSBasis(secs_basis.poles, current_type="unknown")
    with pytest.raises(ValueError, match="current_type"):
        surface_current_matrices(
            evaluation_grid.lat,
            evaluation_grid.lon,
            secs_basis.poles.lat,
            secs_basis.poles.lon,
            current_type="unknown",
        )
    with pytest.raises(ValueError, match="current_type"):
        magnetic_field_matrices(
            evaluation_grid.lat,
            evaluation_grid.lon,
            6371.2e3,
            secs_basis.poles.lat,
            secs_basis.poles.lon,
            current_type="unknown",
        )


@pytest.mark.requires_jax
def test_secs_kernels_preserve_jax_backend_and_numpy_values(secs_basis, evaluation_grid):
    radii = np.array([6371.2e3, 6481.2e3, 6600.0e3, 6371.2e3])
    arguments = (
        evaluation_grid.lat,
        evaluation_grid.lon,
        secs_basis.poles.lat,
        secs_basis.poles.lon,
    )
    with backend_context("numpy"):
        current_reference = surface_current_matrices(
            *arguments, current_type="curl_free", singularity_limit=50e3
        )
        magnetic_reference = magnetic_field_matrices(
            arguments[0],
            arguments[1],
            radii,
            arguments[2],
            arguments[3],
            current_type="divergence_free",
        )

    with backend_context("jax"):
        distances = angular_distance(*arguments)
        current = surface_current_matrices(
            *arguments, current_type="curl_free", singularity_limit=50e3
        )
        magnetic = magnetic_field_matrices(
            arguments[0],
            arguments[1],
            radii,
            arguments[2],
            arguments[3],
            current_type="divergence_free",
        )

    assert "jax" in type(distances).__module__
    for actual, expected in zip(current, current_reference, strict=True):
        assert "jax" in type(actual).__module__
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)
    for actual, expected in zip(magnetic, magnetic_reference, strict=True):
        assert "jax" in type(actual).__module__
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)


@pytest.mark.requires_jax
def test_chunked_secs_operator_is_jittable(secs_basis, evaluation_grid):
    import jax
    import jax.numpy as jnp

    coefficients = jnp.array([0.4, -1.2, 0.7])
    values = jnp.linspace(-1.0, 1.0, 2 * evaluation_grid.size)
    with backend_context("jax"):
        dense = secs_basis.surface_current_operator(evaluation_grid)
        chunked = secs_basis.surface_current_operator(evaluation_grid, chunk_size=2)
        forward = jax.jit(chunked.matvec)(coefficients)
        adjoint = jax.jit(chunked.rmatvec)(values)

    assert "jax" in type(forward).__module__
    assert "jax" in type(adjoint).__module__
    np.testing.assert_allclose(forward, dense @ coefficients, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(adjoint, dense.rmatvec(values), rtol=2e-12, atol=1e-15)


@pytest.mark.requires_jax
def test_current_wedge_kernel_is_jittable_and_matches_numpy():
    import jax
    import jax.numpy as jnp

    arguments = (
        np.array([60.0, 66.0]),
        np.array([0.0, 20.0]),
        np.array([6371.2e3, 6500.0e3]),
        np.array([70.0, 73.0]),
        np.array([5.0, 30.0]),
        np.array([6481.2e3, 6481.2e3]),
        np.array([0.2, -0.1]),
        np.array([0.8, 0.7]),
        np.array([-0.5, -0.6]),
    )
    with backend_context("numpy"):
        reference = current_wedge_magnetic_field_matrices(*arguments)

    jax_arguments = tuple(jnp.asarray(value) for value in arguments)
    with backend_context("jax"):
        actual = jax.jit(current_wedge_magnetic_field_matrices)(*jax_arguments)

    for result, expected in zip(actual, reference, strict=True):
        assert "jax" in type(result).__module__
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=1e-15)
