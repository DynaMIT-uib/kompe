"""Tests for regional cubed-sphere geometry and differential operators."""

from unittest.mock import Mock

import numpy as np
import pytest
from scipy import sparse

from kompe import (
    GlobalCSBasis,
    GlobalCSProjection,
    RegionalCSMesh,
    RegionalCSMeshSpec,
    RegionalCSOperators,
    RegionalCSProjection,
    ScalarBasis,
    SHBasis,
)
from kompe.cubed_sphere import REGIONAL_CS_MESH_SCHEMA, REGIONAL_CS_MESH_SCHEMA_VERSION
from kompe.cubed_sphere.regional_plotting import RegionalCSPlotter


def _longitude_error(actual, expected):
    return (np.asarray(actual) - np.asarray(expected) + 180.0) % 360.0 - 180.0


def test_regional_plotter_uses_grid_resolution_options(monkeypatch):
    axis = Mock()
    grid = Mock(xi_min=-1.0, xi_max=1.0, eta_min=-1.0, eta_max=1.0, radius=6371.2)
    spherical_grid = {}
    km_grid = {}

    monkeypatch.setattr(
        RegionalCSPlotter,
        "add_spherical_grid",
        lambda self, **kwargs: spherical_grid.update(kwargs),
    )
    monkeypatch.setattr(
        RegionalCSPlotter,
        "add_km_grid",
        lambda self, resolution, **kwargs: km_grid.update(resolution=resolution, **kwargs),
    )

    RegionalCSPlotter(axis, grid, gridtype="geo", lat_res=20, lon_res=45)
    RegionalCSPlotter(axis, grid, gridtype="km", km_res=250)

    np.testing.assert_array_equal(spherical_grid["lat_levels"], np.arange(-70, 90, 20))
    np.testing.assert_array_equal(spherical_grid["lon_levels"], np.arange(0, 360, 45))
    assert km_grid["resolution"] == 250


def test_regional_plotter_rejects_unimplemented_grid_options():
    axis = Mock()
    grid = Mock(xi_min=-1.0, xi_max=1.0, eta_min=-1.0, eta_max=1.0)

    with pytest.raises(NotImplementedError, match="Local-time"):
        RegionalCSPlotter(axis, grid, lt=True)
    with pytest.raises(ValueError, match="gridtype"):
        RegionalCSPlotter(axis, grid, gridtype="apex")


def test_regional_plotter_text_can_override_plot_limits(capsys):
    plotter = object.__new__(RegionalCSPlotter)
    plotter.ax = Mock()
    plotter.grid = Mock()
    plotter.grid.projection.geographic_to_cube.return_value = (0.25, -0.5)
    plotter.grid.contains.return_value = False
    plotted_text = object()
    plotter.ax.text.return_value = plotted_text

    assert plotter.text(10.0, 20.0, "outside") is None
    assert "outside plot limit" in capsys.readouterr().out
    assert plotter.text(10.0, 20.0, "outside", ignore_limits=True) is plotted_text
    plotter.ax.text.assert_called_once_with(0.25, -0.5, "outside")


def test_km_grid_samples_each_axis_from_its_own_limits():
    differential_calls = []

    def differential_elements(xi, eta, *steps, **kwargs):
        differential_calls.append((xi, eta))
        return np.ones_like(xi), np.ones_like(eta)

    plotter = object.__new__(RegionalCSPlotter)
    plotter.ax = Mock()
    plotter.grid = Mock(
        xi_min=-0.4,
        xi_max=0.4,
        eta_min=-0.1,
        eta_max=0.1,
        length=1000.0,
        width=400.0,
        radius=6371.2,
    )
    plotter.grid.projection.differential_elements.side_effect = differential_elements

    plotter.add_km_grid(100.0)

    xi_for_eta_gridlines = differential_calls[2][0]
    assert np.min(xi_for_eta_gridlines) == pytest.approx(2 * plotter.grid.xi_min)
    assert np.max(xi_for_eta_gridlines) > 1.9 * plotter.grid.xi_max


@pytest.mark.parametrize("orientation", [0.0, 37.0, [0.6, 0.8]])
def test_projection_roundtrip_is_finite_at_centre_and_axes(orientation):
    projection = RegionalCSProjection((18.0, 67.0), orientation)
    xi = np.array([0.0, 0.0, 0.12, -0.19, 0.21])
    eta = np.array([0.0, 0.18, 0.0, 0.11, -0.16])

    lon, lat = projection.cube_to_geographic(xi, eta)
    actual_xi, actual_eta = projection.geographic_to_cube(lon, lat)

    assert np.isfinite(lon).all()
    assert np.isfinite(lat).all()
    np.testing.assert_allclose(actual_xi, xi, atol=2e-14)
    np.testing.assert_allclose(actual_eta, eta, atol=2e-14)
    centre_lon, centre_lat = projection.cube_to_geographic(np.array(0.0), np.array(0.0))
    np.testing.assert_allclose(_longitude_error(centre_lon, 18.0), 0.0, atol=2e-14)
    np.testing.assert_allclose(centre_lat, 67.0, atol=2e-14)


def test_regional_projection_is_a_rotated_global_north_face():
    projection = RegionalCSProjection((18.0, 67.0), 37.0)
    global_projection = GlobalCSProjection()
    xi = np.array([0.0, 0.12, -0.19, 0.21])
    eta = np.array([0.18, 0.0, 0.11, -0.16])

    lon, lat = projection.cube_to_geographic(xi, eta)
    local_lon, local_lat = projection.geographic_to_local(lon, lat)
    actual_xi, actual_eta = projection.geographic_to_cube(lon, lat)
    expected_xi, expected_eta, face = global_projection.geographic_to_cube(
        local_lon,
        local_lat,
        face=4,
    )

    np.testing.assert_array_equal(face, 4)
    np.testing.assert_allclose(actual_xi, expected_xi, atol=2e-14)
    np.testing.assert_allclose(actual_eta, expected_eta, atol=2e-14)


def test_regional_differential_elements_use_shared_face_metric():
    projection = RegionalCSProjection((18.0, 67.0), 37.0)
    xi = np.array([[0.0], [0.12], [-0.19]])
    eta = np.array([0.18, 0.0, -0.16, 0.07])
    dxi = 0.03
    deta = np.array([0.04, 0.05, 0.06, 0.07])
    radius = 6471.2

    dlxi, dleta, area = projection.differential_elements(
        xi,
        eta,
        dxi,
        deta,
        radius=radius,
    )
    metric = GlobalCSProjection.metric_tensor(xi, eta, radius=radius).reshape(dlxi.shape + (3, 3))

    np.testing.assert_allclose(dlxi, np.sqrt(metric[..., 0, 0]) * dxi)
    np.testing.assert_allclose(dleta, np.sqrt(metric[..., 1, 1]) * deta)
    expected_area = np.sqrt(metric[..., 0, 0] * metric[..., 1, 1] - metric[..., 0, 1] ** 2)
    np.testing.assert_allclose(area, expected_area * dxi * deta)


def test_regional_vector_components_roundtrip_at_face_centre_and_away():
    projection = RegionalCSProjection((18.0, 67.0), 37.0)
    xi = np.array([0.0, 0.07, -0.11, 0.23, -0.19])
    eta = np.array([0.0, -0.08, 0.16, 0.12, -0.21])
    lon, lat = projection.cube_to_geographic(xi, eta)
    east = np.array([1.0, 1.2, -0.7, 0.3, 2.1])
    north = np.array([0.0, -0.4, 0.9, 1.4, -0.2])

    _, _, cube_xi, cube_eta = projection.geographic_vector_to_cube(
        east,
        north,
        lon,
        lat,
    )
    actual_lon, actual_lat, actual_east, actual_north = projection.cube_vector_to_geographic(
        cube_xi,
        cube_eta,
        xi,
        eta,
    )

    assert np.isfinite(cube_xi).all()
    assert np.isfinite(cube_eta).all()
    np.testing.assert_allclose(_longitude_error(actual_lon, lon), 0.0, atol=2e-14)
    np.testing.assert_allclose(actual_lat, lat, atol=2e-14)
    np.testing.assert_allclose(actual_east, east, atol=2e-14)
    np.testing.assert_allclose(actual_north, north, atol=2e-14)


def test_regional_vector_components_match_coordinate_directional_derivative():
    projection = RegionalCSProjection((18.0, 67.0), 37.0)
    xi = np.array([0.0, 0.07, -0.11, 0.23, -0.19])
    eta = np.array([0.0, -0.08, 0.16, 0.12, -0.21])
    lon, lat = projection.cube_to_geographic(xi, eta)
    east = np.array([1.0, 1.2, -0.7, 0.3, 2.1])
    north = np.array([0.0, -0.4, 0.9, 1.4, -0.2])
    magnitude = np.hypot(east, north)

    longitude = np.deg2rad(lon)
    latitude = np.deg2rad(lat)
    position = np.column_stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )
    east_basis = np.column_stack((-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)))
    north_basis = np.column_stack(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        )
    )
    tangent = (east[:, None] * east_basis + north[:, None] * north_basis) / magnitude[:, None]
    step = 1e-6
    displaced = np.cos(step) * position + np.sin(step) * tangent
    displaced_lon = np.rad2deg(np.arctan2(displaced[:, 1], displaced[:, 0]))
    displaced_lat = np.rad2deg(np.arcsin(displaced[:, 2]))
    displaced_xi, displaced_eta = projection.geographic_to_cube(displaced_lon, displaced_lat)
    _, _, cube_xi, cube_eta = projection.geographic_vector_to_cube(east, north, lon, lat)

    np.testing.assert_allclose(cube_xi, (displaced_xi - xi) / step * magnitude, rtol=3e-6)
    np.testing.assert_allclose(cube_eta, (displaced_eta - eta) / step * magnitude, rtol=3e-6)


def test_regional_grid_has_canonical_units_metadata_and_roundtrip():
    projection = RegionalCSProjection((20.0, 70.0), [0.3, 0.7])
    grid = RegionalCSMesh(projection, 1800.0, 1400.0, shape=(18, 14), radius=6371.2)

    assert not isinstance(grid, ScalarBasis)
    assert grid.signature[0] == "REGIONAL_CS_MESH"
    np.testing.assert_allclose(grid.cell_centers.theta, (90.0 - grid.lat).reshape(-1))
    np.testing.assert_allclose(grid.cell_centers.phi, grid.lon.reshape(-1))
    np.testing.assert_allclose(grid.cell_centers.area_weights, grid.cell_areas.reshape(-1))
    assert grid.length == 1800.0
    assert grid.width == 1400.0
    assert grid.shape == (18, 14)
    assert grid.radius == 6371.2
    assert grid.xi_shift == 0.0
    for legacy_name in ("L", "W", "Lres", "Wres", "R", "wshift"):
        assert not hasattr(grid, legacy_name)

    spec = grid.to_spec()
    metadata = spec.to_dict()
    restored = RegionalCSMesh.from_spec(metadata)
    assert metadata["schema"] == REGIONAL_CS_MESH_SCHEMA
    assert metadata["version"] == REGIONAL_CS_MESH_SCHEMA_VERSION
    assert RegionalCSMeshSpec.from_dict(metadata) == spec
    assert restored.signature == grid.signature
    np.testing.assert_allclose(restored.lon, grid.lon)
    np.testing.assert_allclose(restored.lat, grid.lat)

    with pytest.raises(ValueError, match="read-only"):
        grid.lon.flat[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        grid.projection.orientation[0] = 0.0


def test_regional_grid_requires_explicit_radius_and_shifts_only_width_axis():
    projection = RegionalCSProjection((20.0, 70.0), 0.0)
    with pytest.raises(TypeError, match="radius"):
        RegionalCSMesh(projection, 1800.0, 1400.0, shape=(18, 14))

    baseline = RegionalCSMesh(projection, 1800.0, 1400.0, shape=(18, 14), radius=6371.2)
    shifted = RegionalCSMesh(
        projection,
        1800.0,
        1400.0,
        shape=(18, 14),
        radius=6371.2,
        xi_shift=100.0,
    )
    np.testing.assert_allclose(shifted.eta_mesh, baseline.eta_mesh)
    np.testing.assert_allclose(shifted.xi_mesh, baseline.xi_mesh - 100.0 / 6371.2)


def test_regional_grid_names_physical_cell_size_directions():
    projection = RegionalCSProjection((20.0, 70.0), 0.0)
    grid = RegionalCSMesh(
        projection,
        length=1800.0,
        width=1400.0,
        radius=6371.2,
        xi_cell_size=100.0,
        eta_cell_size=200.0,
    )

    assert grid.requested_cell_size == (200.0, 100.0)
    assert grid.n_xi > grid.n_eta
    with pytest.raises(ValueError, match="provided together"):
        RegionalCSMesh(
            projection,
            length=1800.0,
            width=1400.0,
            radius=6371.2,
            xi_cell_size=100.0,
        )


def test_regional_grid_from_edges_preserves_exact_geometry():
    projection = RegionalCSProjection((20.0, 70.0), 0.0)
    xi_edges = np.linspace(-0.18, 0.12, 7)
    eta_edges = np.linspace(-0.10, 0.14, 5)

    grid = RegionalCSMesh.from_edges(
        projection,
        xi_edges,
        eta_edges,
        radius=6371.2,
    )

    np.testing.assert_array_equal(grid.xi_mesh[0], xi_edges)
    np.testing.assert_array_equal(grid.eta_mesh[:, 0], eta_edges)
    restored = RegionalCSMesh.from_spec(grid.to_spec())
    assert restored.signature == grid.signature


def test_version_one_edge_metadata_retains_its_shift_semantics():
    metadata = {
        "schema": REGIONAL_CS_MESH_SCHEMA,
        "version": REGIONAL_CS_MESH_SCHEMA_VERSION,
        "projection": {"position": [20.0, 70.0], "orientation": [1.0, 0.0]},
        "length": 1800.0,
        "width": 1400.0,
        "radius": 6371.2,
        "shape": None,
        "cell_size": None,
        "xi_edges": [-0.1, 0.0, 0.1],
        "eta_edges": [-0.08, 0.0, 0.08],
        "xi_shift": 100.0,
    }

    grid = RegionalCSMesh.from_spec(metadata)

    np.testing.assert_allclose(
        grid.xi_mesh[0],
        np.asarray(metadata["xi_edges"]) - metadata["xi_shift"] / metadata["radius"],
    )
    np.testing.assert_allclose(grid.eta_mesh[:, 0], metadata["eta_edges"])


def test_regional_grid_containment_margin_follows_shifted_boundaries():
    grid = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 0.0),
        1800.0,
        1400.0,
        shape=(18, 14),
        radius=6371.2,
        xi_shift=300.0,
    )
    lon, lat = grid.projection.cube_to_geographic(
        grid.xi_max + grid.dxi / 2,
        (grid.eta_min + grid.eta_max) / 2,
    )

    assert not grid.contains(lon, lat)
    assert grid.contains(lon, lat, margin_cells=1)


def test_regional_grid_rejects_nonuniform_edges_and_invalid_stencils():
    projection = RegionalCSProjection((20.0, 70.0), 0.0)
    with pytest.raises(ValueError, match="uniformly spaced"):
        RegionalCSMesh(
            projection,
            1000.0,
            800.0,
            radius=6371.2,
            xi_edges=[-0.2, -0.1, 0.2],
            eta_edges=[-0.2, 0.0, 0.2],
        )

    grid = RegionalCSMesh(projection, 1000.0, 800.0, shape=(3, 3), radius=6371.2)
    with pytest.raises(ValueError, match="at least"):
        grid.operators.surface_gradient_matrices(stencil_size=2)


def test_regional_scalar_interpolation_preserves_shape_and_complex_values():
    grid = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        shape=(18, 14),
        radius=6371.2,
    )
    values = np.arange(grid.size).reshape(grid.shape) * (1.0 + 2.0j)
    actual = grid.operators.interpolate_scalar(values, grid.lon, grid.lat)

    assert actual.shape == grid.shape
    assert np.iscomplexobj(actual)
    np.testing.assert_allclose(actual, values, atol=1e-10)


def test_regional_scalar_interpolation_is_linear_through_boundary_cells():
    grid = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        shape=(5, 4),
        radius=6371.2,
    )
    values = 2 * grid.xi - 3 * grid.eta
    xi = np.array([grid.xi_min, grid.xi_max])
    eta = np.array([grid.eta_min, grid.eta_max])
    lon, lat = grid.projection.cube_to_geographic(xi, eta)

    actual = grid.operators.interpolate_scalar(values, lon, lat)

    np.testing.assert_allclose(actual, 2 * xi - 3 * eta, atol=2e-14)


def test_regional_scalar_interpolation_supports_singleton_axes():
    grid = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        400.0,
        shape=(1, 4),
        radius=6371.2,
    )
    values = 2 * grid.xi + 5
    lon, lat = grid.projection.cube_to_geographic(grid.xi, grid.eta)

    actual = grid.operators.interpolate_scalar(values, lon, lat)

    np.testing.assert_allclose(actual, values, atol=2e-14)


def test_regional_operator_matrices_are_sparse_by_default():
    grid = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        shape=(18, 14),
        radius=6371.2,
    )
    theta, phi = grid.operators.surface_gradient_matrices()
    divergence = grid.operators.surface_divergence_matrix()

    assert sparse.issparse(theta)
    assert sparse.issparse(phi)
    assert sparse.issparse(divergence)


def test_regional_coordinate_derivatives_are_explicitly_xi_eta():
    grid = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        shape=(18, 14),
        radius=6371.2,
    )
    D_xi, D_eta = grid.operators.coordinate_derivative_matrices()
    xi = grid.xi.reshape(-1)
    eta = grid.eta.reshape(-1)

    np.testing.assert_allclose(D_xi @ xi, 1.0, atol=2e-14)
    np.testing.assert_allclose(D_xi @ eta, 0.0, atol=2e-14)
    np.testing.assert_allclose(D_eta @ xi, 0.0, atol=2e-14)
    np.testing.assert_allclose(D_eta @ eta, 1.0, atol=2e-14)


def test_regional_grid_owns_topology_while_operator_object_owns_numerics():
    grid = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        shape=(18, 14),
        radius=6371.2,
    )
    eta_index = np.array([0, 1, -1])
    xi_index = np.array([0, 2, -1])
    flat = grid.flat_index(eta_index, xi_index)
    actual_eta, actual_xi = grid.unravel_index(flat)

    np.testing.assert_array_equal(actual_eta, eta_index % grid.n_eta)
    np.testing.assert_array_equal(actual_xi, xi_index % grid.n_xi)
    for former_grid_method in (
        "_gradient_matrices",
        "_divergence_matrix",
        "_surface_geometry",
        "_interpolate_scalar",
    ):
        assert not hasattr(grid, former_grid_method)


def test_embedded_metric_matches_cell_area_formula():
    grid = RegionalCSMesh(
        RegionalCSProjection((-25.0, 64.0), [0.8, 0.6]),
        2100.0,
        1300.0,
        shape=(20, 12),
        radius=6471.2,
    )
    assert not hasattr(grid.operators, "surface_geometry")
    _, _, sqrt_g = grid.operators._surface_geometry
    embedded_area = sqrt_g * grid.dxi * grid.deta
    np.testing.assert_allclose(embedded_area, grid.cell_areas.reshape(-1), rtol=6e-15)


def test_regional_gradient_and_divergence_recover_spherical_laplacian():
    grid = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 23.0),
        2000.0,
        1600.0,
        shape=(40, 32),
        radius=6371.2,
    )
    assert isinstance(grid.operators, RegionalCSOperators)
    assert grid.operators is grid.operators
    theta_derivative, phi_derivative = grid.operators.surface_gradient_matrices(
        stencil_size=2, sparse=True
    )
    divergence = grid.operators.surface_divergence_matrix(stencil_size=2, sparse=True)
    assert sparse.issparse(theta_derivative)
    assert sparse.issparse(phi_derivative)
    assert sparse.issparse(divergence)

    # f = sin(latitude) is an l=1 spherical harmonic:
    # grad(f) = -cos(latitude)/R theta and laplacian(f) = -2 f/R^2.
    field = np.sin(np.deg2rad(grid.lat.reshape(-1)))
    expected_theta = -np.cos(np.deg2rad(grid.lat.reshape(-1))) / grid.radius
    theta = theta_derivative @ field
    phi = phi_derivative @ field
    gradient_scale = np.sqrt(np.mean(expected_theta**2))
    gradient_error = np.sqrt(np.mean((theta - expected_theta) ** 2 + phi**2))
    assert gradient_error / gradient_scale < 1e-4

    laplacian = divergence @ np.concatenate([theta, phi])
    expected_laplacian = -2.0 * field / grid.radius**2
    relative_error = np.sqrt(np.mean((laplacian - expected_laplacian) ** 2)) / np.sqrt(
        np.mean(expected_laplacian**2)
    )
    assert relative_error < 5e-4

    gradient_from_operator = grid.operators.surface_gradient_operator(stencil_size=2).matvec(field)
    divergence_from_operator = grid.operators.surface_divergence_operator(stencil_size=2).matvec(
        gradient_from_operator
    )
    np.testing.assert_allclose(
        gradient_from_operator,
        np.concatenate([theta, phi]),
        atol=1e-16,
    )
    np.testing.assert_allclose(divergence_from_operator, laplacian, atol=1e-16)

    # f = x/r = sin(theta) cos(phi) exercises both tangential components.
    colatitude = np.deg2rad(90.0 - grid.lat.reshape(-1))
    longitude = np.deg2rad(grid.lon.reshape(-1))
    field = np.sin(colatitude) * np.cos(longitude)
    expected_theta = np.cos(colatitude) * np.cos(longitude) / grid.radius
    expected_phi = -np.sin(longitude) / grid.radius
    theta = theta_derivative @ field
    phi = phi_derivative @ field
    gradient_error = np.sqrt(np.mean((theta - expected_theta) ** 2 + (phi - expected_phi) ** 2))
    gradient_scale = np.sqrt(np.mean(expected_theta**2 + expected_phi**2))
    assert gradient_error / gradient_scale < 1e-4

    laplacian = divergence @ np.concatenate([theta, phi])
    expected_laplacian = -2.0 * field / grid.radius**2
    relative_error = np.sqrt(np.mean((laplacian - expected_laplacian) ** 2)) / np.sqrt(
        np.mean(expected_laplacian**2)
    )
    assert relative_error < 2e-3


def test_grid_spec_requires_the_versioned_canonical_mapping():
    unversioned = {
        "projection": {"position": [20.0, 70.0], "orientation": [1.0, 0.0]},
        "length": 1800.0,
        "width": 1400.0,
        "shape": [18, 14],
        "radius": 6371.2,
        "xi_shift": 0.0,
    }

    with pytest.raises(ValueError, match="schema"):
        RegionalCSMeshSpec.from_dict(unversioned)


def test_mesh_spec_rejects_unknown_versions_and_ambiguous_construction_modes():
    metadata = {
        "schema": REGIONAL_CS_MESH_SCHEMA,
        "version": REGIONAL_CS_MESH_SCHEMA_VERSION + 1,
        "projection": {"position": [0.0, 60.0], "orientation": [1.0, 0.0]},
        "length": 1000.0,
        "width": 800.0,
        "shape": [10, 8],
        "radius": 6371.2,
    }
    with pytest.raises(ValueError, match="schema version"):
        RegionalCSMeshSpec.from_dict(metadata)

    metadata["version"] = REGIONAL_CS_MESH_SCHEMA_VERSION
    metadata["cell_size"] = [100.0, 100.0]
    with pytest.raises(ValueError, match="exactly one"):
        RegionalCSMeshSpec.from_dict(metadata)


def test_global_and_harmonic_bases_accept_regional_grid_contract():
    grid = RegionalCSMesh(
        RegionalCSProjection((10.0, 60.0), 10.0),
        800.0,
        600.0,
        shape=(6, 4),
        radius=6371.2,
    )
    harmonic = SHBasis(2, 2)
    global_cs = GlobalCSBasis(4)

    assert harmonic.scalar_evaluation_matrix(grid.cell_centers).shape == (
        grid.size,
        harmonic.index_length,
    )
    assert global_cs.scalar_evaluation_matrix(grid.cell_centers).shape == (
        grid.size,
        global_cs.index_length,
    )
