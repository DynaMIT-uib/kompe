"""Tests for the distinction between point grids and structured meshes."""

import numpy as np
import pytest

from kompe import (
    GlobalCSBasis,
    GlobalCSMesh,
    RegionalCSMesh,
    RegionalCSProjection,
    SphericalGrid,
    StructuredSurfaceMesh,
)
from kompe.math import backend_context, get_array_module
from kompe.mesh import spherical_triangle_solid_angle


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_spherical_triangle_geometry_broadcasts_without_a_mesh(backend):
    """Octant triangles have pi/2 steradians regardless of orientation."""
    with backend_context(backend):
        xp = get_array_module()
        first = xp.array([[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]])
        second = xp.array([0.0, 1.0, 0.0])
        third = xp.array([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
        angles = spherical_triangle_solid_angle(first, second, third)
        reversed_angles = spherical_triangle_solid_angle(first, third, second)
        degenerate = spherical_triangle_solid_angle(first, first, third)
        assert isinstance(angles, xp.ndarray)
    assert angles.shape == (2, 2)
    np.testing.assert_allclose(angles, np.pi / 2, rtol=1e-14)
    np.testing.assert_array_equal(reversed_angles, angles)
    np.testing.assert_array_equal(degenerate, np.zeros((2, 2)))


def test_grid_does_not_claim_mesh_topology():
    grid = SphericalGrid(lat=[60.0, 61.0], lon=[10.0, 11.0])

    assert not isinstance(grid, StructuredSurfaceMesh)
    assert grid.shape == (2,)
    assert not hasattr(grid, "cell_areas")
    assert not hasattr(grid, "operators")


def test_global_basis_exposes_validated_native_mesh():
    basis = GlobalCSBasis(4)
    mesh = basis.mesh
    independently_constructed = GlobalCSMesh(4)

    assert isinstance(mesh, GlobalCSMesh)
    assert isinstance(mesh, StructuredSurfaceMesh)
    assert mesh.shape == (6, 4, 4)
    assert mesh.size == basis.index_length == 96
    assert mesh.cell_centers.size == mesh.size
    assert mesh.cell_areas.shape == mesh.shape
    assert mesh.signature == ("GLOBAL_CS_MESH", 4)
    np.testing.assert_allclose(mesh.cell_areas.sum(), 4.0 * np.pi, rtol=2e-15)
    np.testing.assert_array_equal(independently_constructed.theta, mesh.theta)
    np.testing.assert_array_equal(independently_constructed.phi, mesh.phi)
    np.testing.assert_array_equal(independently_constructed.cell_areas, mesh.cell_areas)
    np.testing.assert_array_equal(
        basis.scalar_evaluation_array(mesh.cell_centers), np.eye(mesh.size)
    )


def test_regional_grid_is_a_structured_surface_mesh():
    mesh = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 23.0),
        1800.0,
        1400.0,
        shape=(18, 14),
        radius=6371.2,
    )

    assert isinstance(mesh, StructuredSurfaceMesh)
    assert mesh.shape == (18, 14)
    assert mesh.size == mesh.size
    np.testing.assert_allclose(mesh.cell_centers.theta, (90.0 - mesh.lat).reshape(-1))
    np.testing.assert_allclose(mesh.cell_centers.phi, mesh.lon.reshape(-1))
    np.testing.assert_allclose(mesh.cell_centers.area_weights, mesh.cell_areas.reshape(-1))
    assert np.all(mesh.cell_areas > 0.0)
