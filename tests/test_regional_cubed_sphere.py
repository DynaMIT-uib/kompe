"""Tests for regional cubed-sphere geometry and differential operators."""

import numpy as np
import pytest
from scipy import sparse

from kompe import (
    REGIONAL_CS_GRID_SCHEMA,
    REGIONAL_CS_GRID_SCHEMA_VERSION,
    GlobalCSBasis,
    RegionalCSGrid,
    RegionalCSGridSpec,
    RegionalCSOperators,
    RegionalCSProjection,
    SHBasis,
    SphericalRepresentation,
)


def _longitude_error(actual, expected):
    return (np.asarray(actual) - np.asarray(expected) + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("orientation", [0.0, 37.0, [0.6, 0.8]])
def test_projection_roundtrip_is_finite_at_centre_and_axes(orientation):
    projection = RegionalCSProjection((18.0, 67.0), orientation)
    xi = np.array([0.0, 0.0, 0.12, -0.19, 0.21])
    eta = np.array([0.0, 0.18, 0.0, 0.11, -0.16])

    lon, lat = projection.cube2geo(xi, eta)
    actual_xi, actual_eta = projection.geo2cube(lon, lat)

    assert np.isfinite(lon).all()
    assert np.isfinite(lat).all()
    np.testing.assert_allclose(actual_xi, xi, atol=2e-14)
    np.testing.assert_allclose(actual_eta, eta, atol=2e-14)
    centre_lon, centre_lat = projection.cube2geo(np.array(0.0), np.array(0.0))
    np.testing.assert_allclose(_longitude_error(centre_lon, 18.0), 0.0, atol=2e-14)
    np.testing.assert_allclose(centre_lat, 67.0, atol=2e-14)


def test_regional_grid_has_canonical_units_metadata_and_roundtrip():
    projection = RegionalCSProjection((20.0, 70.0), [0.3, 0.7])
    grid = RegionalCSGrid(projection, 1800.0, 1400.0, 18, 14, radius=6371.2)

    assert isinstance(grid, SphericalRepresentation)
    assert grid.kind == "REGIONAL_CS_GRID"
    assert grid.index_length == grid.size
    assert all(values.size == grid.size for values in grid.index_arrays)
    np.testing.assert_allclose(grid.theta, (90.0 - grid.lat).reshape(-1))
    np.testing.assert_allclose(grid.phi, grid.lon.reshape(-1))
    np.testing.assert_allclose(grid.theta_rad, np.deg2rad(grid.theta))
    np.testing.assert_allclose(grid.phi_rad, np.deg2rad(grid.phi))
    np.testing.assert_allclose(grid.area_weights, grid.A.reshape(-1))
    assert grid.length == 1800.0
    assert grid.width == 1400.0
    assert grid.length_resolution == 18
    assert grid.width_resolution == 14
    assert grid.radius == 6371.2
    assert grid.width_shift == 0.0
    for legacy_name in ("L", "W", "Lres", "Wres", "R", "wshift"):
        assert not hasattr(grid, legacy_name)

    spec = grid.to_spec()
    metadata = spec.to_mapping()
    restored = RegionalCSGrid.from_spec(metadata)
    assert metadata["schema"] == REGIONAL_CS_GRID_SCHEMA
    assert metadata["version"] == REGIONAL_CS_GRID_SCHEMA_VERSION
    assert RegionalCSGridSpec.from_mapping(metadata) == spec
    assert restored.signature == grid.signature
    np.testing.assert_allclose(restored.lon, grid.lon)
    np.testing.assert_allclose(restored.lat, grid.lat)


def test_regional_grid_requires_explicit_radius_and_shifts_only_width_axis():
    projection = RegionalCSProjection((20.0, 70.0), 0.0)
    with pytest.raises(TypeError, match="radius"):
        RegionalCSGrid(projection, 1800.0, 1400.0, 18, 14)

    baseline = RegionalCSGrid(projection, 1800.0, 1400.0, 18, 14, radius=6371.2)
    shifted = RegionalCSGrid(
        projection,
        1800.0,
        1400.0,
        18,
        14,
        radius=6371.2,
        width_shift=100.0,
    )
    np.testing.assert_allclose(shifted.eta_mesh, baseline.eta_mesh)
    np.testing.assert_allclose(shifted.xi_mesh, baseline.xi_mesh - 100.0 / 6371.2)


def test_regional_scalar_interpolation_preserves_shape_and_complex_values():
    grid = RegionalCSGrid(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        18,
        14,
        radius=6371.2,
    )
    values = np.arange(grid.size).reshape(grid.shape) * (1.0 + 2.0j)
    actual = grid.operators.interpolate_scalar(grid.lon, grid.lat, values)

    assert actual.shape == grid.shape
    assert np.iscomplexobj(actual)
    np.testing.assert_allclose(actual, values, atol=1e-10)


def test_regional_operator_matrices_are_sparse_by_default():
    grid = RegionalCSGrid(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        18,
        14,
        radius=6371.2,
    )
    east, north = grid.operators.gradient_matrices()
    divergence = grid.operators.divergence_matrix()

    assert sparse.issparse(east)
    assert sparse.issparse(north)
    assert sparse.issparse(divergence)


def test_embedded_metric_matches_cell_area_formula():
    grid = RegionalCSGrid(
        RegionalCSProjection((-25.0, 64.0), [0.8, 0.6]),
        2100.0,
        1300.0,
        20,
        12,
        radius=6471.2,
    )
    _, _, sqrt_g = grid.operators.surface_geometry()
    embedded_area = sqrt_g * grid.dxi * grid.deta
    np.testing.assert_allclose(embedded_area, grid.A.reshape(-1), rtol=6e-15)


def test_regional_gradient_and_divergence_recover_spherical_laplacian():
    grid = RegionalCSGrid(
        RegionalCSProjection((20.0, 70.0), 23.0),
        2000.0,
        1600.0,
        40,
        32,
        radius=6371.2,
    )
    assert isinstance(grid.operators, RegionalCSOperators)
    assert grid.operators is grid.operators
    east_derivative, north_derivative = grid.operators.gradient_matrices(
        stencil_size=2, sparse=True
    )
    divergence = grid.operators.divergence_matrix(stencil_size=2, sparse=True)
    assert sparse.issparse(east_derivative)
    assert sparse.issparse(north_derivative)
    assert sparse.issparse(divergence)

    # f = sin(latitude) is an l=1 spherical harmonic:
    # grad(f) = cos(latitude)/R north and laplacian(f) = -2 f/R^2.
    field = np.sin(np.deg2rad(grid.lat.reshape(-1)))
    expected_north = np.cos(np.deg2rad(grid.lat.reshape(-1))) / grid.radius
    east = east_derivative @ field
    north = north_derivative @ field
    gradient_scale = np.sqrt(np.mean(expected_north**2))
    gradient_error = np.sqrt(np.mean(east**2 + (north - expected_north) ** 2))
    assert gradient_error / gradient_scale < 1e-4

    laplacian = divergence @ np.concatenate([east, north])
    expected_laplacian = -2.0 * field / grid.radius**2
    relative_error = np.sqrt(np.mean((laplacian - expected_laplacian) ** 2)) / np.sqrt(
        np.mean(expected_laplacian**2)
    )
    assert relative_error < 5e-4


def test_grid_spec_requires_the_versioned_canonical_mapping():
    unversioned = {
        "projection": {"position": [20.0, 70.0], "orientation": [1.0, 0.0]},
        "length": 1800.0,
        "width": 1400.0,
        "length_resolution": 18,
        "width_resolution": 14,
        "radius": 6371.2,
        "width_shift": 0.0,
        "edges": None,
    }

    with pytest.raises(ValueError, match="schema"):
        RegionalCSGridSpec.from_mapping(unversioned)


def test_grid_spec_rejects_unknown_versions_and_mixed_resolution_conventions():
    metadata = {
        "schema": REGIONAL_CS_GRID_SCHEMA,
        "version": REGIONAL_CS_GRID_SCHEMA_VERSION + 1,
        "projection": {"position": [0.0, 60.0], "orientation": [1.0, 0.0]},
        "length": 1000.0,
        "width": 800.0,
        "length_resolution": 10,
        "width_resolution": 8,
        "radius": 6371.2,
    }
    with pytest.raises(ValueError, match="schema version"):
        RegionalCSGridSpec.from_mapping(metadata)

    metadata["version"] = REGIONAL_CS_GRID_SCHEMA_VERSION
    metadata["width_resolution"] = 100.0
    with pytest.raises(ValueError, match="same convention"):
        RegionalCSGridSpec.from_mapping(metadata)


def test_global_and_harmonic_bases_accept_regional_grid_contract():
    grid = RegionalCSGrid(
        RegionalCSProjection((10.0, 60.0), 10.0),
        800.0,
        600.0,
        6,
        4,
        radius=6371.2,
    )
    harmonic = SHBasis(2, 2)
    global_cs = GlobalCSBasis(4)

    assert harmonic.evaluate_on_grid(grid).shape == (grid.size, harmonic.index_length)
    assert global_cs.evaluate_on_grid(grid).shape == (grid.size, global_cs.index_length)
