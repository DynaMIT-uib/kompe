"""Mesh-bound global-CS calculus and geometry-only remapping."""

import numpy as np

from kompe import GlobalCSBasis, GlobalCSMesh, GlobalCSRemapper, SphericalGrid
from kompe.math import get_array_module


def test_mesh_operators_are_shared_by_bases_and_sampled_fields():
    mesh = GlobalCSMesh(8)
    first, second = GlobalCSBasis(mesh=mesh), GlobalCSBasis(mesh=mesh)
    xp = get_array_module()
    values = xp.cos(xp.deg2rad(mesh.theta))
    gradient = mesh.operators.surface_gradient_operator()

    assert first.surface_gradient_operator(mesh.cell_centers) is gradient
    assert second.surface_gradient_operator(mesh.cell_centers) is gradient
    actual = gradient(values)
    assert isinstance(actual, xp.ndarray)
    np.testing.assert_allclose(actual[0], -np.sin(np.deg2rad(mesh.theta)), atol=0.025)
    np.testing.assert_allclose(actual[1], 0, atol=0.025)
    assert not gradient._dense_cache


def test_mesh_poisson_reuses_unit_factors_and_keeps_the_gauge(monkeypatch):
    import kompe.cubed_sphere.global_operators as module

    mesh = GlobalCSMesh(4)
    xp = get_array_module()
    values = xp.asarray(np.random.default_rng(49).normal(size=mesh.size))
    values -= xp.sum(values * xp.asarray(mesh.operators.scalar_mean_weights))
    inverse = mesh.operators.mean_free_surface_poisson_operator()
    expected = inverse(mesh.operators.surface_laplacian_operator()(values))

    def unexpected(*args, **kwargs):
        raise AssertionError("Changing radius must not refactorize the same geometry.")

    monkeypatch.setattr(module, "sparse_least_squares_map", unexpected)
    radius = 3.5
    scaled_inverse = mesh.operators.mean_free_surface_poisson_operator(radius)
    actual = scaled_inverse(mesh.operators.surface_laplacian_operator(radius)(values))
    np.testing.assert_allclose(actual, values, atol=1e-10)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert not inverse._dense_cache
    assert not scaled_inverse._dense_cache


def test_remapping_needs_only_point_grids_not_a_coefficient_basis(monkeypatch):
    source = GlobalCSMesh(8).cell_centers
    target = SphericalGrid(theta=source.theta + 0.02, phi=source.phi + 0.01)
    remapper = GlobalCSRemapper()
    remapper.clear_shared_cache()
    xp = get_array_module()
    values = xp.stack([xp.ones(source.size), xp.ones(source.size) * 2], axis=-1)
    scalar = remapper.scalar_operator(source, target)
    actual = scalar(values)
    np.testing.assert_allclose(actual, values, atol=1e-12)
    assert isinstance(actual, xp.ndarray)
    assert scalar is remapper.scalar_operator(source, target)
    assert not scalar._dense_cache

    independent = GlobalCSRemapper()

    def unexpected(*args, **kwargs):
        raise AssertionError("Identical chart and points must reuse their triangulation.")

    monkeypatch.setattr(independent, "build_scalar_grid_remap_matrix", unexpected)
    np.testing.assert_allclose(independent.scalar_operator(source, target)(values), actual)
    identity = remapper.tangential_operator(source, source)
    vectors = xp.ones((2, source.size, 3))
    np.testing.assert_array_equal(identity(vectors), vectors)
