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


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_secs_angular_distance_resolves_small_separations(backend):
    with backend_context(backend):
        distances = angular_distance([0.0, 0.0], [1e-8, 180.0 - 1e-8], [0.0], [0.0])
    np.testing.assert_allclose(distances[:, 0], np.deg2rad([1e-8, 180.0 - 1e-8]), rtol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
def test_regularized_secs_current_has_zero_limit_at_its_pole(backend, current_type):
    with backend_context(backend), np.errstate(divide="ignore", invalid="ignore"):
        regularized = surface_current_matrices(
            [65.0], [10.0], [65.0], [10.0], current_type=current_type, singularity_limit=1e3
        )
        singular = surface_current_matrices(
            [65.0], [10.0], [65.0], [10.0], current_type=current_type
        )
    np.testing.assert_array_equal(regularized, np.zeros((2, 1, 1)))
    assert not np.isfinite(singular).all()


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_secs_magnetic_field_has_finite_axis_limits(backend):
    from kompe.constants import MU0

    radius = np.array([6371.2e3, 6600e3])
    source_radius = 6481.2e3
    with backend_context(backend), np.errstate(divide="ignore", invalid="ignore"):
        east, north, radial = magnetic_field_matrices(
            [65.0, 65.0], [10.0, 10.0], radius, [65.0], [10.0], source_radius=source_radius
        )
        curl_free_below = magnetic_field_matrices(
            [65.0], [10.0], radius[0], [65.0], [10.0], current_type="curl_free"
        )
        curl_free_regularized = magnetic_field_matrices(
            [65.0, 65.0],
            [10.0, 10.0],
            radius,
            [65.0],
            [10.0],
            current_type="curl_free",
            singularity_limit=1e3,
        )
    s = np.minimum(radius, source_radius) / np.maximum(radius, source_radius)
    expected = MU0 / (4 * np.pi * radius) * (1 / (1 - s) - 1)
    expected[1] *= s[1]
    np.testing.assert_array_equal(east, np.zeros((2, 1)))
    np.testing.assert_array_equal(north, np.zeros((2, 1)))
    np.testing.assert_allclose(radial[:, 0], expected, rtol=1e-13)
    np.testing.assert_array_equal(curl_free_below, np.zeros((3, 1, 1)))
    np.testing.assert_array_equal(curl_free_regularized, np.zeros((3, 2, 1)))


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_secs_magnetic_field_resolves_the_near_axis_slope(backend):
    from kompe.constants import MU0

    radius = np.array([6371.2e3, 6600e3])
    source_radius = 6481.2e3
    angle = np.deg2rad(1e-8)
    with backend_context(backend), np.errstate(divide="ignore", invalid="ignore"):
        east, north, _ = magnetic_field_matrices(
            [0.0, 0.0], [1e-8, 1e-8], radius, [0.0], [0.0], source_radius=source_radius
        )
    s = np.minimum(radius, source_radius) / np.maximum(radius, source_radius)
    expected_slope = MU0 / (4 * np.pi * radius) / (2 * (1 - s) ** 2)
    expected_slope *= [-s[0] * (2 - s[0]), s[1] ** 2]
    np.testing.assert_allclose(east[:, 0] / angle, expected_slope, rtol=1e-12)
    np.testing.assert_allclose(north, 0.0, atol=1e-30)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_secs_field_matches_published_formulas_away_from_the_axis(backend):
    from kompe.constants import MU0

    source_radius = 6481.2e3
    theta, radius_ratio = np.meshgrid([0.2, 0.7, 1.5, 2.5, 3.0], [0.1, 0.5, 0.9, 1, 1.1, 2, 10])
    theta, radius_ratio = theta.reshape(-1), radius_ratio.reshape(-1)
    radius = source_radius * radius_ratio
    with backend_context(backend):
        east, north, radial = magnetic_field_matrices(
            90 - np.rad2deg(theta),
            np.zeros(theta.size),
            radius,
            [90.0],
            [0.0],
            source_radius=source_radius,
        )
    # Vanhamaki and Juusola (2020), equations 2.13--2.14, with the
    # source at the north pole: local poleward direction is geographic north.
    s = np.minimum(radius_ratio, 1) / np.maximum(radius_ratio, 1)
    root = np.sqrt(1 + s**2 - 2 * s * np.cos(theta))
    factor = MU0 / (4 * np.pi * radius)
    expected_radial = factor * np.where(radius_ratio <= 1, 1 / root - 1, s / root - s)
    expected_north = (
        factor
        / np.sin(theta)
        * np.where(
            radius_ratio <= 1,
            (s - np.cos(theta)) / root + np.cos(theta),
            (1 - s * np.cos(theta)) / root - 1,
        )
    )
    np.testing.assert_allclose(east, 0.0, atol=1e-27)
    np.testing.assert_allclose(north[:, 0], expected_north, rtol=1e-11, atol=1e-27)
    np.testing.assert_allclose(radial[:, 0], expected_radial, rtol=1e-12, atol=1e-27)


def test_secs_is_scalar_synthesis_without_closed_surface_claims(secs_basis):
    assert isinstance(secs_basis, ScalarBasis)
    assert not isinstance(secs_basis, SurfaceDifferentialBasis)
    assert isinstance(secs_basis, SECSBasis)
    assert secs_basis.kind == "SECS"
    assert secs_basis.coefficient_count == 3
    assert secs_basis.index_names == ("latitude", "longitude")


def test_secs_accepts_regional_grid_for_poles_and_evaluation():
    regional = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 25.0),
        600.0,
        500.0,
        shape=(4, 3),
        radius=6371.2,
    )
    basis = SECSBasis(poles=regional.cell_centers, current_type="curl_free")
    array = basis.scalar_evaluation_array(regional)

    assert basis.coefficient_count == regional.size
    assert array.shape == (regional.size, regional.size)


def test_secs_scalar_synthesis_has_explicit_physical_mode(evaluation_grid):
    poles = SphericalGrid(lat=[65.0, 70.0], lon=[5.0, 18.0])
    curl_free = SECSBasis(poles=poles, current_type="curl_free")
    divergence_free = SECSBasis(poles=poles, current_type="divergence_free")

    expected_potential = scalar_green_matrix(
        evaluation_grid.lat,
        evaluation_grid.lon,
        poles.lat,
        poles.lon,
        quantity="curl_free_potential",
        normalization=curl_free.normalization,
    )
    expected_df_potential = scalar_green_matrix(
        evaluation_grid.lat,
        evaluation_grid.lon,
        poles.lat,
        poles.lon,
        quantity="divergence_free_potential",
        normalization=divergence_free.normalization,
    )

    np.testing.assert_allclose(
        curl_free.scalar_evaluation_array(evaluation_grid), expected_potential
    )
    np.testing.assert_allclose(
        divergence_free.scalar_evaluation_array(evaluation_grid), expected_df_potential
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
    canonical = basis.surface_current_array(evaluation_grid)

    np.testing.assert_allclose(canonical[0], -north)
    np.testing.assert_allclose(canonical[1], east)
    operator = basis.surface_current_operator(evaluation_grid)
    coefficients = np.array([0.4, -1.2, 0.7])
    np.testing.assert_allclose(
        operator @ coefficients,
        np.tensordot(canonical, coefficients, axes=1).reshape(-1),
    )


def test_two_component_secs_helmholtz_operator_matches_array(secs_basis, evaluation_grid):
    array = secs_basis.helmholtz_current_synthesis_array(evaluation_grid)
    operator = secs_basis.helmholtz_current_synthesis_operator(evaluation_grid)
    coefficients = np.array([[0.2, -0.5, 0.8], [1.0, 0.3, -0.4]])

    assert array.shape == (2, evaluation_grid.size, 2, secs_basis.coefficient_count)
    np.testing.assert_allclose(
        operator @ coefficients.reshape(-1),
        np.tensordot(array, coefficients, axes=2).reshape(-1),
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
    canonical = basis.magnetic_field_array(evaluation_grid, evaluation_radius)
    np.testing.assert_allclose(canonical, np.stack([radial, -north, east]))


def test_secs_kernel_does_not_mutate_numpy_error_policy(secs_basis, evaluation_grid):
    before = np.geterr().copy()
    secs_basis.magnetic_field_array(evaluation_grid, 6371.2e3)
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
