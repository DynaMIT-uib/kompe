"""Tests for the distinction between point grids and structured meshes."""

import numpy as np

from kompe import (
    GlobalCSBasis,
    GlobalCSMesh,
    Grid,
    RegionalCSGrid,
    RegionalCSProjection,
    StructuredSurfaceMesh,
)


def test_grid_does_not_claim_mesh_topology():
    grid = Grid(lat=[60.0, 61.0], lon=[10.0, 11.0])

    assert not isinstance(grid, StructuredSurfaceMesh)
    assert not hasattr(grid, "mesh_shape")


def test_global_basis_exposes_validated_native_mesh():
    basis = GlobalCSBasis(4)
    mesh = basis.mesh
    independently_constructed = GlobalCSMesh(4)

    assert isinstance(mesh, GlobalCSMesh)
    assert isinstance(mesh, StructuredSurfaceMesh)
    assert basis.grid_geometry is mesh
    assert mesh.mesh_shape == (6, 4, 4)
    assert mesh.cell_count == basis.index_length == 96
    assert mesh.cell_center_theta.shape == mesh.mesh_shape
    assert mesh.cell_center_phi.shape == mesh.mesh_shape
    assert mesh.cell_areas.shape == mesh.mesh_shape
    assert mesh.signature == ("GLOBAL_CS_MESH", 4)
    assert mesh.size == mesh.cell_count
    np.testing.assert_allclose(mesh.cell_areas.sum(), 4.0 * np.pi, rtol=2e-15)
    np.testing.assert_array_equal(independently_constructed.arr_theta, mesh.arr_theta)
    np.testing.assert_array_equal(independently_constructed.arr_phi, mesh.arr_phi)
    np.testing.assert_array_equal(independently_constructed.cell_areas, mesh.cell_areas)
    np.testing.assert_array_equal(basis.evaluate_on_grid(mesh), np.eye(mesh.cell_count))


def test_regional_grid_is_a_structured_surface_mesh():
    mesh = RegionalCSGrid(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        18,
        14,
        radius=6371.2,
    )

    assert isinstance(mesh, StructuredSurfaceMesh)
    assert mesh.mesh_shape == (18, 14)
    assert mesh.cell_count == mesh.size
    np.testing.assert_allclose(mesh.cell_center_theta, 90.0 - mesh.lat)
    np.testing.assert_allclose(mesh.cell_center_phi, mesh.lon)
    np.testing.assert_allclose(mesh.cell_areas, mesh.A)
    assert np.all(mesh.cell_areas > 0.0)
