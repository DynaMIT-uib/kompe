"""Tests for transforms between spherical representations."""

import numpy as np
import pytest

from kompe import GlobalCSBasis, SHBasis, SphericalGrid, SphericalTransform
from kompe.cubed_sphere.global_remapping import _GlobalCSRemapper
from kompe.math import get_backend, jax_enabled, set_backend, to_numpy
from kompe.math.least_squares_solver import dense_full_rank_least_squares_map


def _regular_grid():
    lat = np.linspace(-70.0, 70.0, 11)
    lon = np.linspace(0.0, 330.0, 12)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return SphericalGrid(lat=lat_grid.reshape(-1), lon=lon_grid.reshape(-1))


# Scalar SH analysis and transform caches


def test_transform_cache_controls_rebuild_equivalent_analysis():
    """Transform-local cached state can be inspected and discarded."""
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid())
    values = np.linspace(-1.0, 1.0, transform.grid.size)
    expected = transform.analyze_scalar(values)

    assert transform.cache_info()["scalar_factorization"]
    assert transform.cache_info()["materialized_values"] > 0
    transform.clear_cache()
    assert not transform.cache_info()["scalar_factorization"]
    assert transform.cache_info()["materialized_values"] == 0
    np.testing.assert_allclose(transform.analyze_scalar(values), expected)


def test_spherical_transform_analyzes_scalar_grid_values():
    """Scalar analysis recovers known coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    expected = np.zeros(basis.index_length)
    expected[1] = 1.0
    expected[3] = -0.25
    values = transform.synthesize_scalar(expected)

    actual = transform.analyze_scalar_samples(values, input_grid=grid)

    np.testing.assert_allclose(actual[0], expected, atol=1e-10)


def test_transform_with_basis_reuses_compatible_and_cached_transforms():
    """One grid can serve multiple coefficient spaces without caller-side caches."""
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=0.1, area_weighted=True)

    assert transform.with_basis(SHBasis(3, 2, mean_free=True)) is transform

    other_basis = SHBasis(4, 2, mean_free=True)
    rebound = transform.with_basis(other_basis)

    assert rebound is transform.with_basis(other_basis)
    assert rebound.basis is other_basis
    assert rebound.grid is transform.grid
    assert rebound.reg_lambda == transform.reg_lambda
    assert rebound.area_weighted == transform.area_weighted
    assert transform.cache_info()["basis_transforms"] == 1

    transform.clear_cache()
    assert transform.cache_info()["basis_transforms"] == 0

    with pytest.raises(TypeError, match="SurfaceDifferentialBasis"):
        transform.with_basis(object())


def test_explicit_empty_solver_name_is_not_treated_as_default():
    """An invalid explicit solver selection fails visibly."""
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid())

    with pytest.raises(ValueError, match="Solver must be one of"):
        transform.analyze_scalar(np.zeros(transform.grid.size), solver_type="")


def test_rotated_gradient_analysis_matches_dense_least_squares():
    """Structured potential analysis preserves the dense definition."""
    basis = SHBasis(3, 3, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid, area_weighted=True)
    scale = np.linspace(0.5, 1.5, basis.index_length)
    synthesis = (
        np.asarray(transform.rhat_cross_gradient_matrix) * scale.reshape(1, 1, -1)
    ).reshape(2 * grid.size, basis.index_length)
    expected = dense_full_rank_least_squares_map(
        synthesis,
        sqrt_weights=transform.helmholtz_sqrt_weights,
        input_shape=(2, grid.size),
        output_shape=(basis.index_length,),
    )

    observed = transform.rhat_cross_gradient_analysis_operator(coefficient_scale=scale)

    np.testing.assert_allclose(
        observed.to_matrix(backend="numpy"),
        expected.to_matrix(backend="numpy"),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("field_type", ["scalar", "helmholtz"])
def test_structured_sh_normal_matrix_matches_explicit_system(field_type):
    """Memory-bounded SH normals preserve the explicit system."""
    basis = SHBasis(3, 3, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=0.2, area_weighted=True)
    problem = (
        transform.scalar_least_squares_problem
        if field_type == "scalar"
        else transform.helmholtz_least_squares_problem
    )

    observed = np.asarray(problem.dense_normal_matrix())
    system = np.asarray(problem.system_operator.to_matrix(backend="numpy"))

    np.testing.assert_allclose(observed, system.T @ system, rtol=1e-12, atol=1e-12)


def test_spherical_transform_caches_sample_analysis_transforms_by_grid():
    """Direct analysis cache keeps separate compatible input grids."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    shifted_grid = SphericalGrid(lat=grid.lat, lon=grid.lon + 1.0)
    transform = SphericalTransform(basis, grid)

    transform.analyze_scalar_samples(np.zeros(grid.size), input_grid=grid, analysis_basis=basis)
    first_cached = tuple(transform._analysis_transforms.values())
    transform.analyze_scalar_samples(
        np.zeros(shifted_grid.size), input_grid=shifted_grid, analysis_basis=basis
    )
    transform.analyze_scalar_samples(np.zeros(grid.size), input_grid=grid, analysis_basis=basis)

    assert len(transform._analysis_transforms) == 2
    assert tuple(transform._analysis_transforms.values())[-1] is first_cached[0]


def test_weighted_analysis_cache_distinguishes_grid_measures():
    """Different grid measures use different analyses."""
    basis = SHBasis(3, 2, mean_free=True)
    target = _regular_grid()
    first = SphericalGrid(lat=target.lat, lon=target.lon, area_weights=np.ones(target.size))
    second = SphericalGrid(
        lat=target.lat, lon=target.lon, area_weights=np.linspace(1.0, 2.0, target.size)
    )
    transform = SphericalTransform(basis, target, area_weighted=True)
    values = np.zeros(target.size)

    assert first.same_as(second)
    transform.analyze_scalar_samples(values, input_grid=first, analysis_basis=basis)
    transform.analyze_scalar_samples(values, input_grid=second, analysis_basis=basis)

    assert len(transform._analysis_transforms) == 2


def test_direct_analysis_cache_fingerprints_explicit_weights():
    """Equal explicit weights reuse one immutable analysis transform."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    baseline_weights = np.linspace(0.5, 1.0, grid.size)
    supplied_weights = baseline_weights.copy()
    values = np.zeros(grid.size)

    transform.analyze_scalar_samples(
        values,
        input_grid=grid,
        analysis_basis=basis,
        sqrt_weights=supplied_weights,
        reg_lambda=0.1,
    )
    cached_transform = next(iter(transform._analysis_transforms.values()))
    supplied_weights[0] = 2.0

    transform.analyze_scalar_samples(
        values,
        input_grid=grid,
        analysis_basis=basis,
        sqrt_weights=baseline_weights.copy(),
        reg_lambda=0.1,
    )

    assert len(transform._analysis_transforms) == 1
    assert next(iter(transform._analysis_transforms.values())) is cached_transform
    if isinstance(cached_transform.sqrt_weights, np.ndarray):
        assert not cached_transform.sqrt_weights.flags.writeable
    else:
        assert "jax" in type(cached_transform.sqrt_weights).__module__
    np.testing.assert_array_equal(cached_transform.sqrt_weights, baseline_weights)

    transform.analyze_scalar_samples(
        values,
        input_grid=grid,
        analysis_basis=basis,
        sqrt_weights=supplied_weights,
        reg_lambda=0.1,
    )

    assert len(transform._analysis_transforms) == 2


def test_direct_analysis_cache_treats_zero_regularization_as_none():
    """Equivalent unregularized requests share one analysis transform."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    values = np.zeros(grid.size)

    transform.analyze_scalar_samples(values, input_grid=grid, analysis_basis=basis, reg_lambda=0.0)
    transform.analyze_scalar_samples(
        values, input_grid=grid, analysis_basis=basis, reg_lambda=None
    )

    assert len(transform._analysis_transforms) == 1


# Spectral regularization


def test_spherical_transform_regularization_uses_diagonal_operators():
    """Keep surface smoothness structured in least-squares."""
    basis = SHBasis(4, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=1.0)
    n = np.asarray(basis.n, dtype=float)
    q = 1.0 / (2.0 * n + 1.0)
    mu = n * (n + 1.0)
    scalar_weights = np.sqrt(q * mu)
    helmholtz_weights = np.broadcast_to(np.sqrt(q) * mu, (2, basis.index_length))
    helmholtz_coeffs = np.vstack(
        [np.linspace(0.0, 1.0, basis.index_length), np.linspace(1.0, 2.0, basis.index_length)]
    )
    scalar_coeffs = np.linspace(0.0, 1.0, basis.index_length)

    scalar_regularization = transform.scalar_least_squares_problem.regularization_operators[0]
    helmholtz_regularization = transform.helmholtz_least_squares_problem.regularization_operators[
        0
    ]

    np.testing.assert_allclose(scalar_regularization.diagonal(backend="numpy"), scalar_weights)
    np.testing.assert_allclose(
        helmholtz_regularization.diagonal(backend="numpy"), helmholtz_weights.reshape(-1)
    )
    np.testing.assert_allclose(
        transform.apply_helmholtz_regularization(helmholtz_coeffs),
        helmholtz_weights * helmholtz_coeffs,
    )
    np.testing.assert_allclose(
        transform.apply_scalar_regularization(scalar_coeffs), scalar_weights * scalar_coeffs
    )


def test_zero_regularization_uses_unregularized_analysis_path():
    """Zero regularization does not disable structured analysis."""
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=0.0)

    assert transform.reg_lambda is None
    assert transform.helmholtz_analysis_operator is not None


@pytest.mark.parametrize("reg_lambda", [-1.0, np.inf, np.nan])
def test_transform_rejects_invalid_regularization(reg_lambda):
    """Invalid regularization fails at transform construction."""
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        SphericalTransform(SHBasis(3, 2), _regular_grid(), reg_lambda=reg_lambda)


def test_surface_smoothness_regularization_matches_parseval_weights():
    """Regularizer norms reproduce analytic Schmidt-basis energies."""
    basis = SHBasis(5, 3, mean_free=False)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=1.0)
    rng = np.random.default_rng(17)
    scalar = rng.normal(size=basis.index_length)
    helmholtz = rng.normal(size=(2, basis.index_length))
    n = np.asarray(basis.n, dtype=float)
    q = 1.0 / (2.0 * n + 1.0)
    mu = n * (n + 1.0)

    scalar_penalty = np.linalg.norm(transform.apply_scalar_regularization(scalar)) ** 2
    vector_penalty = np.linalg.norm(transform.apply_helmholtz_regularization(helmholtz)) ** 2

    np.testing.assert_allclose(scalar_penalty, np.sum(q * mu * scalar**2))
    np.testing.assert_allclose(vector_penalty, np.sum(q * mu**2 * helmholtz**2))
    assert transform.scalar_regularization_operator.diagonal(backend="numpy")[0] == 0.0
    np.testing.assert_allclose(
        transform.helmholtz_regularization_operator.diagonal(backend="numpy").reshape(
            2, basis.index_length
        )[0],
        transform.helmholtz_regularization_operator.diagonal(backend="numpy").reshape(
            2, basis.index_length
        )[1],
    )


def test_schmidt_surface_norms_match_gauss_legendre_quadrature():
    """Match analytic regularizer normalization to the SH basis."""
    latitude_nodes, latitude_weights = np.polynomial.legendre.leggauss(16)
    theta = np.rad2deg(np.arccos(latitude_nodes))
    phi = np.linspace(0.0, 360.0, 33, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    solid_angle = np.broadcast_to(
        latitude_weights[:, None] * (2.0 * np.pi / phi.size), theta_grid.shape
    )
    grid = SphericalGrid(
        theta=theta_grid.reshape(-1),
        phi=phi_grid.reshape(-1),
        area_weights=solid_angle.reshape(-1),
    )
    basis = SHBasis(5, 5, mean_free=False)
    values = np.asarray(basis.scalar_evaluation_matrix(grid))
    gradient = np.asarray(basis.surface_gradient_matrix(grid))
    normalized_area = solid_angle.reshape(-1) / (4.0 * np.pi)
    q = 1.0 / (2.0 * basis.n + 1.0)
    mu = basis.n * (basis.n + 1.0)

    value_norms = np.sum(normalized_area[:, None] * values**2, axis=0)
    gradient_norms = np.sum(normalized_area[None, :, None] * gradient**2, axis=(0, 1))

    np.testing.assert_allclose(value_norms, q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient_norms, q * mu, rtol=1e-12, atol=1e-12)


def test_unnormalized_sh_smoothness_matches_gauss_legendre_quadrature():
    """Regularization follows the selected SH coefficient normalization."""
    latitude_nodes, latitude_weights = np.polynomial.legendre.leggauss(16)
    theta = np.rad2deg(np.arccos(latitude_nodes))
    phi = np.linspace(0.0, 360.0, 33, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    solid_angle = np.broadcast_to(
        latitude_weights[:, None] * (2.0 * np.pi / phi.size), theta_grid.shape
    )
    grid = SphericalGrid(theta=theta_grid, phi=phi_grid)
    basis = SHBasis(5, 5, mean_free=False, schmidt_quasi_normalized=False)
    gradient = np.asarray(basis.surface_gradient_matrix(grid))
    normalized_area = solid_angle.reshape(-1) / (4.0 * np.pi)
    gradient_norms = np.sum(normalized_area[None, :, None] * gradient**2, axis=(0, 1))

    np.testing.assert_allclose(
        gradient_norms,
        basis.scalar_smoothness_weights() ** 2,
        rtol=1e-12,
        atol=1e-12,
    )


def test_scalar_smoothness_is_invariant_to_log_reference_mode():
    """A logarithmic reference change affects only the free mean."""
    basis = SHBasis(5, 3, mean_free=False)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=1.0)
    coefficients = np.linspace(-1.0, 1.0, basis.index_length)
    shifted = coefficients.copy()
    shifted[np.asarray(basis.n) == 0] += 7.5

    np.testing.assert_allclose(
        transform.apply_scalar_regularization(shifted),
        transform.apply_scalar_regularization(coefficients),
    )


def test_spherical_transform_requires_configured_regularization():
    """Missing regularization configuration produces a clear error."""
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid())

    with pytest.raises(RuntimeError, match="Scalar regularization requires reg_lambda"):
        transform.apply_scalar_regularization(np.zeros(basis.index_length))
    with pytest.raises(RuntimeError, match="Helmholtz regularization requires reg_lambda"):
        transform.apply_helmholtz_regularization(np.zeros((2, basis.index_length)))


def test_regularized_transform_does_not_expose_unregularized_analysis_operator():
    """A fixed analysis map cannot omit configured regularization."""
    transform = SphericalTransform(SHBasis(3, 2, mean_free=True), _regular_grid(), reg_lambda=1.0)

    with pytest.raises(RuntimeError, match="only available for unregularized"):
        _ = transform.helmholtz_analysis_operator


# Tangential and batched analysis


def test_spherical_transform_analyzes_tangential_grid_values():
    """Tangential analysis recovers Helmholtz coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    expected = np.zeros((2, basis.index_length))
    expected[0, 1] = 1.0
    expected[1, 3] = -0.5
    values = transform.synthesize_helmholtz(expected)

    actual = transform.analyze_helmholtz_samples(values, input_grid=grid, analysis_basis=basis)
    direct = transform.analyze_helmholtz(values)

    np.testing.assert_allclose(actual[0], expected.reshape(-1), atol=1e-10)
    np.testing.assert_allclose(direct, expected, atol=1e-10)
    assert "helmholtz_analysis_operator" not in transform.__dict__


def test_spherical_transform_batches_direct_analysis():
    """Direct SH analysis handles multiple RHS columns at once."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    scalar_coeffs = np.zeros((2, basis.index_length))
    scalar_coeffs[0, 1] = 1.0
    scalar_coeffs[1, 3] = -0.25
    scalar_values = np.vstack([transform.synthesize_scalar(row) for row in scalar_coeffs])
    vector_coeffs = np.zeros((2, 2, basis.index_length))
    vector_coeffs[0, 0, 1] = 1.0
    vector_coeffs[1, 1, 3] = -0.5
    vector_values = np.stack([transform.synthesize_helmholtz(row) for row in vector_coeffs])

    scalar_actual = transform.analyze_scalar_samples(
        scalar_values, input_grid=grid, analysis_basis=basis
    )
    vector_actual = transform.analyze_helmholtz_samples(
        vector_values, input_grid=grid, analysis_basis=basis
    )

    np.testing.assert_allclose(scalar_actual, scalar_coeffs, atol=1e-10)
    np.testing.assert_allclose(vector_actual, vector_coeffs.reshape(2, -1), atol=1e-10)


def test_spherical_transform_least_squares_use_operator_properties():
    """Least-squares setup should not force dense attributes."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)

    scalar_problem = transform.scalar_least_squares_problem
    helmholtz_problem = transform.helmholtz_least_squares_problem

    assert scalar_problem.A[0] is transform.scalar_synthesis_operator
    assert helmholtz_problem.A[0] is transform.helmholtz_synthesis_operator
    assert "scalar_synthesis_matrix" not in transform.__dict__
    assert "helmholtz_synthesis_matrix" not in transform.__dict__


# Cubed-sphere remapping and analysis


def test_native_cs_transform_synthesizes_from_sparse_operator_paths(monkeypatch):
    """Native CS synthesis can apply sparse operators."""
    basis = GlobalCSBasis(4)
    grid = SphericalGrid(
        theta=basis.mesh.theta, phi=basis.mesh.phi, area_weights=basis.mesh.cell_areas.reshape(-1)
    )
    transform = SphericalTransform(basis, grid)
    derivatives = basis._native_derivatives
    theta = derivatives["theta"].toarray()
    phi = derivatives["phi"].toarray()

    scalar_coeffs = np.linspace(0.0, 1.0, basis.index_length)
    vector_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])
    expected_helmholtz = np.stack(
        [
            -theta @ vector_coeffs[0] - phi @ vector_coeffs[1],
            -phi @ vector_coeffs[0] + theta @ vector_coeffs[1],
        ]
    )

    def fail_evaluate_on_grid(*args, **kwargs):
        raise AssertionError("native CS synthesis should use operator paths")

    monkeypatch.setattr(basis, "scalar_evaluation_matrix", fail_evaluate_on_grid)

    np.testing.assert_allclose(transform.synthesize_scalar(scalar_coeffs), scalar_coeffs)
    np.testing.assert_allclose(
        transform.synthesize_scalar(scalar_coeffs, derivative="theta"), theta @ scalar_coeffs
    )
    np.testing.assert_allclose(
        transform.synthesize_scalar(scalar_coeffs, derivative="phi"), phi @ scalar_coeffs
    )
    np.testing.assert_allclose(transform.synthesize_helmholtz(vector_coeffs), expected_helmholtz)
    assert "scalar_synthesis_matrix" not in transform.__dict__
    assert "helmholtz_synthesis_matrix" not in transform.__dict__


def test_spherical_transform_reuses_scalar_grid_remap(monkeypatch):
    """Scalar analysis reuses a cached CS remap operator."""
    _GlobalCSRemapper._shared_remap_matrix_cache.clear()
    basis = SHBasis(3, 2, mean_free=True)
    remapping_basis = GlobalCSBasis(8)
    source_basis = GlobalCSBasis(10)
    target_grid = SphericalGrid(
        theta=remapping_basis.mesh.theta,
        phi=remapping_basis.mesh.phi,
        area_weights=remapping_basis.mesh.cell_areas.reshape(-1),
    )
    input_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    values = np.vstack([np.sin(np.deg2rad(input_grid.theta)), np.cos(np.deg2rad(input_grid.phi))])
    transform = SphericalTransform(basis, target_grid)
    calls = 0
    original = remapping_basis._remapper.build_scalar_grid_remap_matrix

    def counted_build_scalar_grid_remap_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        remapping_basis._remapper,
        "build_scalar_grid_remap_matrix",
        counted_build_scalar_grid_remap_matrix,
    )

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("supported CS remaps should use cached operators")

    monkeypatch.setattr(remapping_basis, "interpolate_scalar", fail_interpolate_scalar)

    projected_1 = transform.analyze_scalar_samples(
        values, input_grid=input_grid, analysis_basis=remapping_basis
    )
    projected_2 = transform.analyze_scalar_samples(
        values, input_grid=input_grid, analysis_basis=remapping_basis
    )

    assert calls == 1
    assert projected_1.shape == (2, basis.index_length)
    np.testing.assert_allclose(projected_2, projected_1)


def test_spherical_transform_skips_matching_grid_remap(monkeypatch):
    """Sample analysis skips remapping on matching grids."""
    basis = SHBasis(3, 2, mean_free=True)
    remapping_basis = GlobalCSBasis(8)
    grid = SphericalGrid(
        theta=remapping_basis.mesh.theta,
        phi=remapping_basis.mesh.phi,
        area_weights=remapping_basis.mesh.cell_areas.reshape(-1),
    )
    values = np.vstack([np.sin(np.deg2rad(grid.theta)), np.cos(np.deg2rad(grid.phi))])
    transform = SphericalTransform(basis, grid)

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("matching grids should not interpolate")

    monkeypatch.setattr(remapping_basis, "interpolate_scalar", fail_interpolate_scalar)

    projected = transform.analyze_scalar_samples(
        values, input_grid=grid, analysis_basis=remapping_basis
    )

    assert projected.shape == (2, basis.index_length)


def test_remapped_sample_analysis_applies_target_fit_options(monkeypatch):
    """Regularization and tolerance configure the post-remap analysis."""
    basis = SHBasis(3, 2, mean_free=True)
    remapping_basis = GlobalCSBasis(6)
    source_basis = GlobalCSBasis(8)
    input_grid = source_basis.native_grid
    values = np.sin(np.deg2rad(input_grid.theta))
    transform = SphericalTransform(basis, remapping_basis.native_grid)
    recorded = {}
    original = transform._sample_analysis_transform

    def record_analysis_transform(*args, **kwargs):
        recorded.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(transform, "_sample_analysis_transform", record_analysis_transform)
    projected = transform.analyze_scalar_samples(
        values,
        input_grid=input_grid,
        analysis_basis=remapping_basis,
        reg_lambda=1e-3,
        pinv_rtol=1e-10,
    )

    assert projected.shape == (1, basis.index_length)
    assert recorded["reg_lambda"] == 1e-3
    assert recorded["pinv_rtol"] == 1e-10


def test_remapped_sample_analysis_rejects_source_grid_weights():
    """Input-grid weights are not silently reused after interpolation."""
    basis = SHBasis(3, 2, mean_free=True)
    remapping_basis = GlobalCSBasis(6)
    source_basis = GlobalCSBasis(8)
    transform = SphericalTransform(basis, remapping_basis.native_grid)
    input_grid = source_basis.native_grid
    values = np.sin(np.deg2rad(input_grid.theta))

    with pytest.raises(ValueError, match="cannot be propagated through grid remapping"):
        transform.analyze_scalar_samples(
            values,
            input_grid=input_grid,
            analysis_basis=remapping_basis,
            sqrt_weights=np.ones(input_grid.size),
        )


def test_spherical_transform_requires_grid_remap_operator():
    """SphericalGrid-to-grid analysis requires remap operators."""
    basis = SHBasis(3, 2, mean_free=True)
    target_basis = GlobalCSBasis(8)
    source_basis = GlobalCSBasis(10)
    target_grid = SphericalGrid(
        theta=target_basis.mesh.theta,
        phi=target_basis.mesh.phi,
        area_weights=target_basis.mesh.cell_areas.reshape(-1),
    )
    input_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    values = np.sin(np.deg2rad(input_grid.theta))
    target_basis.scalar_grid_remap_operator = None
    transform = SphericalTransform(basis, target_grid)

    with pytest.raises(TypeError, match="scalar_grid_remap_operator"):
        transform.analyze_scalar_samples(
            values, input_grid=input_grid, analysis_basis=target_basis
        )


def test_spherical_transform_reuses_helmholtz_grid_remap(monkeypatch):
    """Helmholtz analysis reuses a cached CS remap operator."""
    _GlobalCSRemapper._shared_remap_matrix_cache.clear()
    basis = SHBasis(3, 2, mean_free=True)
    remapping_basis = GlobalCSBasis(8)
    source_basis = GlobalCSBasis(10)
    target_grid = SphericalGrid(
        theta=remapping_basis.mesh.theta,
        phi=remapping_basis.mesh.phi,
        area_weights=remapping_basis.mesh.cell_areas.reshape(-1),
    )
    input_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    theta_values = np.vstack(
        [np.sin(np.deg2rad(input_grid.theta)), np.cos(np.deg2rad(input_grid.theta))]
    )
    phi_values = np.vstack(
        [np.cos(np.deg2rad(input_grid.phi)), np.sin(np.deg2rad(input_grid.phi))]
    )
    values = np.stack([theta_values, phi_values], axis=1)
    transform = SphericalTransform(basis, target_grid)
    calls = 0
    original = remapping_basis._remapper.build_tangential_grid_remap_matrix

    def counted_build_tangential_grid_remap_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        remapping_basis._remapper,
        "build_tangential_grid_remap_matrix",
        counted_build_tangential_grid_remap_matrix,
    )

    def fail_interpolate_vector(*args, **kwargs):
        raise AssertionError("supported CS remaps should use cached operators")

    monkeypatch.setattr(remapping_basis, "interpolate_vector", fail_interpolate_vector)

    projected_1 = transform.analyze_helmholtz_samples(
        values, input_grid=input_grid, analysis_basis=remapping_basis
    )
    projected_2 = transform.analyze_helmholtz_samples(
        values, input_grid=input_grid, analysis_basis=remapping_basis
    )

    assert calls == 1
    assert projected_1.shape == (2, 2 * basis.index_length)
    np.testing.assert_allclose(projected_2, projected_1)


def test_cs_scalar_remap_operator_matches_interpolation():
    """Cached scalar remap matches the legacy CS interpolation."""
    source_basis = GlobalCSBasis(8)
    target_basis = GlobalCSBasis(6)
    source_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    target_grid = SphericalGrid(theta=target_basis.mesh.theta, phi=target_basis.mesh.phi)
    values = np.sin(np.deg2rad(source_grid.theta)) + 0.25 * np.cos(np.deg2rad(source_grid.phi))

    operator = target_basis.scalar_grid_remap_operator(source_grid, target_grid)
    actual = operator @ values
    expected = target_basis.interpolate_scalar(
        values, source_grid.theta, source_grid.phi, target_grid.theta, target_grid.phi
    )

    assert operator is target_basis.scalar_grid_remap_operator(source_grid, target_grid)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_cs_tangential_remap_operator_matches_interpolation():
    """Cached tangential remap matches legacy interpolation."""
    source_basis = GlobalCSBasis(8)
    target_basis = GlobalCSBasis(6)
    source_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    target_grid = SphericalGrid(theta=target_basis.mesh.theta, phi=target_basis.mesh.phi)
    theta_component = np.sin(np.deg2rad(source_grid.theta))
    phi_component = np.cos(np.deg2rad(source_grid.phi))
    values = np.vstack([theta_component, phi_component])

    operator = target_basis.tangential_grid_remap_operator(source_grid, target_grid)
    actual = (operator @ values.reshape(-1)).reshape(2, target_grid.size)
    expected_theta, expected_phi, _ = target_basis.interpolate_vector(
        theta_component,
        phi_component,
        np.zeros_like(theta_component),
        source_grid.theta,
        source_grid.phi,
        target_grid.theta,
        target_grid.phi,
    )
    expected = np.vstack([expected_theta, expected_phi])

    assert operator is target_basis.tangential_grid_remap_operator(source_grid, target_grid)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_cs_tangential_remap_matrix_cache_is_shared(monkeypatch):
    """Equivalent CS remaps share sparse matrix construction."""
    _GlobalCSRemapper._shared_remap_matrix_cache.clear()
    source_basis = GlobalCSBasis(8)
    target_basis = GlobalCSBasis(6)
    equivalent_target_basis = GlobalCSBasis(6)
    source_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    target_grid = SphericalGrid(theta=target_basis.mesh.theta, phi=target_basis.mesh.phi)
    values = np.vstack(
        [np.sin(np.deg2rad(source_grid.theta)), np.cos(np.deg2rad(source_grid.phi))]
    )

    first_operator = target_basis.tangential_grid_remap_operator(source_grid, target_grid)

    def fail_build(*args, **kwargs):
        raise AssertionError("equivalent remap matrix should come from shared cache")

    monkeypatch.setattr(
        equivalent_target_basis._remapper, "build_tangential_grid_remap_matrix", fail_build
    )

    second_operator = equivalent_target_basis.tangential_grid_remap_operator(
        source_grid, target_grid
    )

    np.testing.assert_allclose(
        second_operator @ values.reshape(-1), first_operator @ values.reshape(-1)
    )


def test_cs_non_native_scalar_operator_uses_remap_without_dense_interpolation(monkeypatch):
    """CS non-native scalar operators use sparse remaps."""
    basis = GlobalCSBasis(8)
    _, theta, phi = basis.mesh.projection.cube_to_spherical(
        basis.mesh.face_coordinate(np.array([1.2, 2.3, 3.4, 4.5])),
        basis.mesh.face_coordinate(np.array([1.1, 2.2, 3.1, 4.2])),
        np.zeros(4),
        degrees=True,
    )
    target = SphericalGrid(theta=theta, phi=phi)
    coeffs = np.sin(np.deg2rad(basis.mesh.theta)) + 0.25 * np.cos(np.deg2rad(basis.mesh.phi))
    expected = basis.interpolate_scalar(
        coeffs, basis.mesh.theta, basis.mesh.phi, target.theta, target.phi
    )

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("scalar operator should use the remap LinearMap path")

    monkeypatch.setattr(basis, "interpolate_scalar", fail_interpolate_scalar)

    operator = basis.scalar_evaluation_operator(target)

    assert operator.output_shape == (target.size,)
    np.testing.assert_allclose(operator.matvec(coeffs), expected, atol=1e-12)


def test_cs_non_native_vector_operators_use_remap_without_dense_interpolation(monkeypatch):
    """CS non-native vector operators use sparse remaps."""
    basis = GlobalCSBasis(8)
    _, theta, phi = basis.mesh.projection.cube_to_spherical(
        basis.mesh.face_coordinate(np.array([1.2, 2.3, 3.4, 4.5])),
        basis.mesh.face_coordinate(np.array([1.1, 2.2, 3.1, 4.2])),
        np.zeros(4),
        degrees=True,
    )
    target = SphericalGrid(theta=theta, phi=phi)
    scalar_coeffs = np.sin(np.deg2rad(basis.mesh.theta)) + 0.25 * np.cos(
        np.deg2rad(basis.mesh.phi)
    )
    helmholtz_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])

    expected_gradient = np.tensordot(basis.surface_gradient_matrix(target), scalar_coeffs, axes=1)
    expected_rxgrad = np.tensordot(basis.rhat_cross_gradient_matrix(target), scalar_coeffs, axes=1)
    expected_helmholtz = np.tensordot(
        basis.helmholtz_synthesis_matrix(target), helmholtz_coeffs, axes=2
    )

    def fail_interpolate_vector(*args, **kwargs):
        raise AssertionError("vector operator should use the remap LinearMap path")

    monkeypatch.setattr(basis, "interpolate_vector", fail_interpolate_vector)

    gradient_operator = basis.surface_gradient_operator(target)
    rxgrad_operator = basis.rhat_cross_gradient_operator(target)
    helmholtz_operator = basis.helmholtz_synthesis_operator(target)

    np.testing.assert_allclose(
        gradient_operator.matvec(scalar_coeffs).reshape(2, target.size),
        expected_gradient,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rxgrad_operator.matvec(scalar_coeffs).reshape(2, target.size), expected_rxgrad, atol=1e-10
    )
    np.testing.assert_allclose(
        helmholtz_operator.matvec(helmholtz_coeffs.reshape(-1)).reshape(2, target.size),
        expected_helmholtz,
        atol=1e-10,
    )


def test_cs_non_native_scalar_analysis_solves_against_remap_operator():
    """CS scalar analysis is identity only on the native grid."""
    basis = GlobalCSBasis(4)
    target_basis = GlobalCSBasis(6)
    target = SphericalGrid(
        theta=target_basis.mesh.theta,
        phi=target_basis.mesh.phi,
        area_weights=target_basis.mesh.cell_areas.reshape(-1),
    )
    transform = SphericalTransform(basis, target)
    coeff_rows = np.vstack(
        [np.sin(np.deg2rad(basis.mesh.theta)), np.cos(np.deg2rad(basis.mesh.phi))]
    )
    value_rows = np.stack([transform.synthesize_scalar(row) for row in coeff_rows])

    coeffs = transform.analyze_scalar(value_rows[0])
    projected_rows = transform.analyze_scalar_samples(
        value_rows, input_grid=target, analysis_basis=basis
    )

    assert coeffs.shape == (basis.index_length,)
    assert projected_rows.shape == coeff_rows.shape
    np.testing.assert_allclose(transform.synthesize_scalar(coeffs), value_rows[0])
    for projected, expected_values in zip(projected_rows, value_rows, strict=True):
        np.testing.assert_allclose(transform.synthesize_scalar(projected), expected_values)


def test_direct_sample_analysis_uses_the_analysis_transform_layout():
    """A native direct analysis owns the returned batch layout."""

    class DirectNativeCSBasis(GlobalCSBasis):
        sample_analysis_uses_grid_remapping = False

    basis = DirectNativeCSBasis(4)
    target = SphericalGrid(theta=basis.mesh.theta + 1e-3, phi=basis.mesh.phi)
    transform = SphericalTransform(basis, target)
    values = np.vstack([np.arange(basis.index_length), -np.arange(basis.index_length)])

    projected = transform.analyze_scalar_samples(
        values, input_grid=basis.native_grid, analysis_basis=basis
    )

    np.testing.assert_array_equal(projected, values)


def test_native_grid_identity_analysis_respects_explicit_zero_weights():
    """Identity synthesis still solves when explicit weights remove data."""
    basis = GlobalCSBasis(4)
    weights = np.ones(basis.native_grid.size)
    weights[0] = 0.0
    values = np.linspace(-1.0, 1.0, basis.native_grid.size)
    transform = SphericalTransform(basis, basis.native_grid, sqrt_weights=weights)

    analyzed = transform.analyze_scalar(values)

    expected = values.copy()
    expected[0] = 0.0
    np.testing.assert_allclose(analyzed, expected)


def test_cs_non_native_helmholtz_analysis_solves_against_remap_operator():
    """CS Helmholtz analysis is identity only on the native grid."""
    basis = GlobalCSBasis(4)
    target_basis = GlobalCSBasis(6)
    target = SphericalGrid(
        theta=target_basis.mesh.theta,
        phi=target_basis.mesh.phi,
        area_weights=target_basis.mesh.cell_areas.reshape(-1),
    )
    transform = SphericalTransform(basis, target)
    base = np.sin(np.deg2rad(basis.mesh.theta)) + 0.25 * np.cos(np.deg2rad(basis.mesh.phi))
    coeffs = basis.project_helmholtz_mean_free(np.vstack([base, base[::-1]]))
    values = transform.synthesize_helmholtz(coeffs)

    actual = transform.analyze_helmholtz(values)

    assert actual.shape == (2, basis.index_length)
    np.testing.assert_allclose(transform.synthesize_helmholtz(actual), values, atol=1e-10)


# Optimized Helmholtz analysis


@pytest.mark.parametrize("area_weighted", [False, True])
def test_mean_free_sh_helmholtz_analysis_uses_full_rank_factorization(area_weighted):
    """Gauge-free SH analysis avoids a tall SVD on either backend."""
    basis = SHBasis(4, 3, mean_free=True)
    cs_basis = GlobalCSBasis(8)
    grid = SphericalGrid(
        theta=cs_basis.mesh.theta,
        phi=cs_basis.mesh.phi,
        area_weights=cs_basis.mesh.cell_areas.reshape(-1),
    )
    transform = SphericalTransform(basis, grid, area_weighted=area_weighted)
    reference = SphericalTransform(basis, grid, area_weighted=area_weighted)
    rng = np.random.default_rng(20260718)
    values = rng.normal(size=(2, grid.size))

    operator = transform.helmholtz_analysis_operator
    actual = transform.analyze_helmholtz(values)
    expected = reference.analyze_helmholtz(values, solver_type="normal_pinv")

    assert operator._cached_dense(np) is None
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    if jax_enabled():
        import jax
        import jax.numpy as jnp

        assert "jax" in type(actual).__module__
        compiled = jax.jit(transform.analyze_helmholtz)(jnp.asarray(values))
        np.testing.assert_allclose(compiled, expected, rtol=2e-12, atol=2e-12)


def test_full_mean_sh_helmholtz_analysis_retains_rank_deficient_fallback():
    """Constant SH gauges continue through pseudoinverse analysis."""
    basis = SHBasis(3, 2, mean_free=False)
    cs_basis = GlobalCSBasis(6)
    grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
    transform = SphericalTransform(basis, grid)

    assert transform._optimized_helmholtz_analysis_operator is None
    assert transform.helmholtz_analysis_operator.shape == (2 * basis.index_length, 2 * grid.size)


def test_optimized_helmholtz_analysis_rejects_ambiguous_value_layout():
    """A matching element count does not define component and sample axes."""
    basis = GlobalCSBasis(4)
    grid = SphericalGrid(theta=basis.mesh.theta, phi=basis.mesh.phi)
    transform = SphericalTransform(basis, grid)

    with pytest.raises(ValueError, match="incompatible with data_shape"):
        transform.analyze_helmholtz(np.ones((4, grid.size)))


def test_helmholtz_factorization_does_not_hide_invalid_weights():
    """Only rank deficiency may select the pseudoinverse fallback."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid, sqrt_weights=np.ones(grid.size - 1))

    with pytest.raises(ValueError, match="Helmholtz sqrt_weights"):
        _ = transform._optimized_helmholtz_analysis_operator


def test_helmholtz_point_weights_apply_to_both_components():
    """One explicit weight per point has the same meaning for theta and phi."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    point_weights = np.linspace(0.5, 1.5, grid.size)
    component_weights = np.tile(point_weights, (2, 1))
    values = np.vstack([np.sin(np.deg2rad(grid.theta)), np.cos(np.deg2rad(grid.phi))])

    point_transform = SphericalTransform(basis, grid, sqrt_weights=point_weights)
    component_transform = SphericalTransform(basis, grid, sqrt_weights=component_weights)

    np.testing.assert_allclose(point_transform.helmholtz_sqrt_weights, component_weights)
    np.testing.assert_allclose(
        point_transform.analyze_helmholtz(values),
        component_transform.analyze_helmholtz(values),
    )


@pytest.mark.parametrize("area_weighted", [False, True])
def test_native_cs_helmholtz_analysis_is_sparse_constrained_least_squares(area_weighted):
    """Native CS analysis stays sparse and fixes both gauges."""
    basis = GlobalCSBasis(4)
    grid = SphericalGrid(
        theta=basis.mesh.theta, phi=basis.mesh.phi, area_weights=basis.mesh.cell_areas.reshape(-1)
    )
    transform = SphericalTransform(basis, grid, area_weighted=area_weighted)
    reference_transform = SphericalTransform(basis, grid, area_weighted=area_weighted)
    rng = np.random.default_rng(42)
    values = rng.normal(size=(2, grid.size))

    operator = transform.helmholtz_analysis_operator
    actual = operator.matvec(values).reshape(2, basis.index_length)
    api_actual = transform.analyze_helmholtz(values)
    expected = reference_transform.analyze_helmholtz(values, solver_type="normal_pinv")
    expected = basis.project_helmholtz_mean_free(expected)

    assert operator._cached_dense(np) is None
    assert "helmholtz_least_squares_problem" not in transform.__dict__
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(api_actual, actual)
    np.testing.assert_allclose(basis.scalar_mean(actual), np.zeros(2), atol=2e-14)
    np.testing.assert_allclose(
        transform.synthesize_helmholtz(actual),
        transform.synthesize_helmholtz(expected),
        rtol=2e-11,
        atol=2e-11,
    )

    coefficient_probe = rng.normal(size=operator.shape[0])
    grid_probe = rng.normal(size=operator.shape[1])
    np.testing.assert_allclose(
        np.vdot(coefficient_probe, operator.matvec(grid_probe)),
        np.vdot(operator.rmatvec(coefficient_probe), grid_probe),
        rtol=2e-12,
        atol=2e-12,
    )

    value_batch = rng.normal(size=(3, 2, grid.size))
    batch_actual = transform.analyze_helmholtz(value_batch)
    batch_expected = np.stack(
        [
            basis.project_helmholtz_mean_free(
                reference_transform.analyze_helmholtz(row, solver_type="normal_pinv")
            )
            for row in value_batch
        ],
        axis=-1,
    )
    np.testing.assert_allclose(batch_actual, batch_expected, rtol=2e-11, atol=2e-11)

    if jax_enabled():
        import jax
        import jax.numpy as jnp

        assert "jax" in type(api_actual).__module__
        compiled = jax.jit(transform.analyze_helmholtz)(jnp.asarray(values))
        np.testing.assert_allclose(compiled, actual, rtol=2e-11, atol=2e-11)


# Backend preservation


@pytest.mark.requires_jax
def test_cs_grid_remap_geometry_is_built_on_numpy(monkeypatch):
    """SciPy triangulation should use NumPy cube-coordinate geometry."""
    import kompe.cubed_sphere.global_remapping as remapping_module

    _GlobalCSRemapper.clear_shared_cache()
    remapping_basis = GlobalCSBasis(8)
    source_basis = GlobalCSBasis(10)
    original_delaunay = remapping_module.Delaunay
    observed_backends = []

    def checked_delaunay(*args, **kwargs):
        observed_backends.append(get_backend())
        return original_delaunay(*args, **kwargs)

    previous_backend = jax_enabled()
    try:
        set_backend("jax")
        monkeypatch.setattr(remapping_module, "Delaunay", checked_delaunay)
        remapping_basis.scalar_grid_remap_operator(
            source_basis.native_grid, remapping_basis.native_grid
        )
        assert get_backend() == "jax"
    finally:
        set_backend(previous_backend)

    assert observed_backends
    assert set(observed_backends) == {"numpy"}


@pytest.mark.requires_jax
def test_spherical_transform_synthesis_preserves_jax_backend():
    """Coefficient-to-grid synthesis uses LinearMap backend handling."""
    previous_backend = jax_enabled()
    try:
        set_backend("jax")
        basis = GlobalCSBasis(4)
        grid = SphericalGrid(
            theta=basis.mesh.theta,
            phi=basis.mesh.phi,
            area_weights=basis.mesh.cell_areas.reshape(-1),
        )

        transform = SphericalTransform(basis, grid)
        scalar_coeffs = np.linspace(0.0, 1.0, basis.index_length)
        scalar_values = transform.synthesize_scalar(scalar_coeffs)
        assert "jax" in type(scalar_values).__module__
        backend_dtype = to_numpy(scalar_values).dtype
        assert np.issubdtype(backend_dtype, np.floating)
        assert to_numpy(transform.scalar_synthesis_matrix).dtype == backend_dtype
        np.testing.assert_allclose(
            to_numpy(scalar_values), to_numpy(transform.scalar_synthesis_matrix) @ scalar_coeffs
        )

        vector_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])
        vector_values = transform.synthesize_helmholtz(vector_coeffs)
        assert "jax" in type(vector_values).__module__
        assert to_numpy(vector_values).dtype == backend_dtype
        assert to_numpy(transform.helmholtz_synthesis_matrix).dtype == backend_dtype
        np.testing.assert_allclose(
            to_numpy(vector_values),
            np.tensordot(to_numpy(transform.helmholtz_synthesis_matrix), vector_coeffs, 2),
        )
    finally:
        set_backend(previous_backend)


@pytest.mark.requires_jax
def test_spherical_transform_preserves_explicit_jax_coefficients():
    """Explicit JAX coefficients reach the LinearMap apply path."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    try:
        set_backend("numpy")
        basis = GlobalCSBasis(4)
        grid = SphericalGrid(
            theta=np.asarray(basis.mesh.theta),
            phi=np.asarray(basis.mesh.phi),
            area_weights=np.asarray(basis.mesh.cell_areas.reshape(-1)),
        )
        transform = SphericalTransform(basis, grid)
        coeffs = jnp.linspace(0.0, 1.0, basis.index_length)

        values = transform.synthesize_scalar(coeffs)

        assert "jax" in type(values).__module__
        np.testing.assert_allclose(
            to_numpy(values), transform.scalar_synthesis_matrix @ to_numpy(coeffs)
        )
    finally:
        set_backend(previous_backend)
