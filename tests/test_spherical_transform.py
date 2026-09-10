"""Tests for transforms between spherical representations."""

import numpy as np
import pytest

import kompe.spherical_transform as spherical_transform_module
from kompe import (
    GlobalCSBasis,
    SHBasis,
    SphericalGrid,
    SphericalTransform,
)
from kompe.cubed_sphere.global_remapping import GlobalCSRemapper
from kompe.math import (
    LeastSquaresSolver,
    LinearMap,
    as_linear_map,
    backend_context,
    get_array_module,
    get_backend,
    identity_linear_map,
    jax_enabled,
    set_backend,
    to_numpy,
)
from kompe.math.least_squares_solver import dense_full_rank_least_squares_map


@pytest.mark.parametrize("sampling", ["off_grid", "permuted_basis"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("tolerance", [1e-12, 0.1])
def test_fixed_helmholtz_analysis_preserves_physical_gauges(sampling, weighted, tolerance):
    """Fixed and configurable analysis use the same constrained SVD objective."""
    from scipy.linalg import null_space

    from kompe.basis import BasisSubset

    basis = GlobalCSBasis(4)
    grid = basis.native_grid
    if sampling == "off_grid":
        grid = SphericalGrid(theta=grid.theta, phi=grid.phi + 0.3)
    else:
        basis = BasisSubset(basis, np.arange(basis.coefficient_count)[::-1])
    rng = np.random.default_rng(128)
    weights = rng.uniform(0.4, 1.5, (2, grid.size)) if weighted else None
    transform = SphericalTransform(basis, grid, sqrt_weights=weights, tolerance=tolerance)
    samples = get_array_module().asarray(rng.normal(size=(2, grid.size, 2, 3)))
    synthesis = basis.helmholtz_synthesis_operator(grid).to_matrix(backend="numpy")
    Z = null_space(basis.helmholtz_gauge_constraints)
    w = np.ones(2 * grid.size) if weights is None else weights.reshape(-1)
    reference = Z @ np.linalg.pinv(w[:, None] * (synthesis @ Z), rtol=tolerance) * w
    expected = (reference @ np.asarray(samples).reshape(2 * grid.size, -1)).reshape(samples.shape)

    inverse = transform.helmholtz_analysis_operator
    actual = inverse(samples)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.einsum("n,cnij->cij", basis.scalar_mean_weights, np.asarray(actual)), 0, atol=1e-13
    )
    fitted = transform.analyze_helmholtz(samples, LeastSquaresSolver("svd", tolerance=tolerance))
    np.testing.assert_allclose(actual, fitted, rtol=1e-10, atol=1e-12)
    coefficients = rng.normal(size=inverse.output_shape)
    lhs = np.vdot(np.asarray(inverse(samples[..., 0, 0])), coefficients)
    rhs = np.vdot(np.asarray(samples[..., 0, 0]), np.asarray(inverse.adjoint()(coefficients)))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("array_first", [False, True])
def test_sh_helmholtz_array_and_operator_share_materialization(array_first, monkeypatch):
    """Array inspection and operator application share one dense result."""
    from kompe.basis import SurfaceDifferentialBasis

    basis = SHBasis(3, 2)
    grid = SphericalGrid(lat=[30.0, 45.0, 60.0], lon=[0.0, 45.0, 90.0])
    transform = SphericalTransform(basis, grid)
    operator = basis.helmholtz_synthesis_operator(grid)
    assert not operator._dense_cache
    theta = np.asarray(basis.scalar_evaluation_array(grid, "theta"))
    phi = np.asarray(basis.scalar_evaluation_array(grid, "phi"))
    expected = np.block([[-theta, -phi], [-phi, theta]]).reshape(
        2, grid.size, 2, basis.coefficient_count
    )
    if array_first:
        array = basis.helmholtz_synthesis_array(grid)
        matrix = operator.to_matrix()
    else:
        matrix = operator.to_matrix()
        array = basis.helmholtz_synthesis_array(grid)

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("The existing Helmholtz materialization must be reused.")

    monkeypatch.setattr(SurfaceDifferentialBasis, "helmholtz_synthesis_array", unexpected_rebuild)
    assert operator.to_matrix() is matrix
    assert basis.helmholtz_synthesis_array(grid) is array
    np.testing.assert_allclose(array, expected)
    np.testing.assert_allclose(transform.helmholtz_synthesis_array, expected)
    if get_array_module() is np:
        assert np.shares_memory(array, matrix)
        assert np.shares_memory(transform.helmholtz_synthesis_array, matrix)


def test_basis_compatibility_distinguishes_equal_coefficient_counts():
    """Array lengths alone cannot identify coefficient spaces."""
    source = SHBasis(1, 1)
    different = SHBasis(3, 0)
    assert source.coefficient_count == different.coefficient_count
    assert not source.coefficients_are_compatible_with(different)
    assert source.coefficients_are_compatible_with(SHBasis(1, 1))


@pytest.mark.parametrize("representation", ["scalar", "helmholtz"])
def test_synthesis_and_analysis_share_trailing_batch_axes(representation):
    """The output of batched analysis can be synthesized directly."""
    xp = get_array_module()
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid())
    coefficient_shape = (
        (basis.coefficient_count,) if representation == "scalar" else (2, basis.coefficient_count)
    )
    rng = np.random.default_rng(14)
    coefficients = xp.asarray(rng.normal(size=coefficient_shape + (2, 3)))
    synthesize = getattr(transform, f"synthesize_{representation}")
    analyze = getattr(transform, f"analyze_{representation}")

    values = synthesize(coefficients)

    expected = np.stack(
        [synthesize(coefficients[..., i, j]) for i in range(2) for j in range(3)], axis=-1
    ).reshape(values.shape)
    np.testing.assert_allclose(values, expected, atol=1e-12)
    recovered = analyze(values)
    assert recovered.shape == coefficients.shape
    np.testing.assert_allclose(recovered, coefficients, atol=1e-10)
    np.testing.assert_allclose(synthesize(recovered), values, atol=1e-10)


def test_scalar_synthesis_rejects_unknown_gradient_component():
    """An unsupported component must not evaluate the undifferentiated field."""
    basis = SHBasis(2, 1)
    transform = SphericalTransform(basis, SphericalGrid(lat=[30.0], lon=[0.0]))
    with pytest.raises(ValueError, match="gradient_component"):
        transform.synthesize_scalar(np.ones(basis.coefficient_count), gradient_component="radial")


def test_named_gradient_components_include_the_spherical_metric():
    """For f = sin(theta) sin(phi), grad_phi f = cos(phi), including poles."""
    basis = SHBasis(1, 1, mean_free=True)
    grid = _regular_grid()
    samples = np.sin(np.deg2rad(grid.theta)) * np.sin(np.deg2rad(grid.phi))
    coefficients = SphericalTransform(basis, grid).analyze_scalar(samples)
    target = SphericalGrid(
        theta=[0.0, 30.0, 75.0, 150.0, 180.0], phi=[20.0, 40.0, 60.0, 80.0, 100.0]
    )
    transform = SphericalTransform(basis, target)
    theta, phi = np.deg2rad(target.theta), np.deg2rad(target.phi)
    expected = {"theta": np.cos(theta) * np.sin(phi), "phi": np.cos(phi)}
    for component, values in expected.items():
        operator = getattr(transform, f"gradient_{component}_operator")
        array = getattr(transform, f"gradient_{component}_array")
        np.testing.assert_allclose(operator(coefficients), values, atol=1e-12)
        np.testing.assert_allclose(array @ coefficients, values, atol=1e-12)
        np.testing.assert_allclose(
            transform.synthesize_scalar(coefficients, gradient_component=component),
            values,
            atol=1e-12,
        )


@pytest.mark.parametrize("field", ["scalar", "helmholtz"])
@pytest.mark.parametrize("solver", ["normal_pinv", "normal_solve"])
def test_analysis_requires_leading_data_axes(field, solver):
    """Time-first provider arrays must be converted at the input boundary."""
    grid = _regular_grid()
    transform = SphericalTransform(SHBasis(3, 2), grid)
    shape = (2, grid.size) if field == "helmholtz" else (grid.size,)
    values = np.zeros((3, *shape))
    with pytest.raises(ValueError, match="data_shape"):
        getattr(transform, f"analyze_{field}")(values, solver=solver)
    with pytest.raises(ValueError, match="data_shape"):
        getattr(transform, f"analyze_{field}_samples")(values, input_grid=grid, solver=solver)


@pytest.mark.parametrize("helmholtz", [False, True])
@pytest.mark.parametrize("kind", ["diagonal", "sparse", "matrix_free"])
@pytest.mark.parametrize("materialized", [False, True])
def test_explicit_sample_remap_preserves_structure(helmholtz, kind, materialized):
    """Even on identical grids, apply the requested map and retain its storage."""
    from scipy.sparse import diags

    xp = get_array_module()
    grid = _regular_grid()
    transform = SphericalTransform(SHBasis(3, 2), grid)
    shape = (2, grid.size) if helmholtz else (grid.size,)
    diagonal = xp.linspace(0.4, 1.3, int(np.prod(shape)))
    if kind == "matrix_free":
        operator = LinearMap(
            shape=(diagonal.size, diagonal.size),
            dtype=diagonal.dtype,
            input_shape=shape,
            output_shape=shape,
            matvec=lambda x: diagonal * x,
            rmatvec=lambda x: diagonal * x,
            matmat=lambda x: diagonal[:, None] * x,
            rmatmat=lambda x: diagonal[:, None] * x,
            dense_array=lambda xp: xp.diag(xp.asarray(diagonal)),
            backend_operands=(diagonal,),
        )
    else:
        array = diags(to_numpy(diagonal), format="csr") if kind == "sparse" else diagonal
        operator = as_linear_map(array, input_shape=shape, output_shape=shape)
    if materialized:
        operator.to_matrix()
    values = xp.asarray(np.random.default_rng(92).normal(size=shape + (2, 3)))
    field = "helmholtz" if helmholtz else "scalar"
    actual = getattr(transform, f"analyze_{field}_samples")(
        values, input_grid=grid, remap=operator
    )
    expected = getattr(transform, f"analyze_{field}")(diagonal.reshape(shape + (1, 1)) * values)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    assert bool(operator._dense_cache) == materialized


def _regular_grid():
    lat = np.linspace(-70.0, 70.0, 11)
    lon = np.linspace(0.0, 330.0, 12)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return SphericalGrid(lat=lat_grid, lon=lon_grid, area_weights=np.cos(np.deg2rad(lat_grid)))


# Scalar SH analysis and transform caches


def test_transform_cache_controls_rebuild_equivalent_analysis():
    """Transform-local cached state can be inspected and discarded."""
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid())
    values = np.linspace(-1.0, 1.0, transform.grid.size)
    expected = transform.analyze_scalar(values)

    assert transform.cache_info()["scalar_problem_cached"]
    assert transform.cache_info()["cached_attributes"] > 0
    transform.clear_cache()
    assert not transform.cache_info()["scalar_problem_cached"]
    assert transform.cache_info()["cached_attributes"] == 0
    np.testing.assert_allclose(transform.analyze_scalar(values), expected)


def test_spherical_transform_analyzes_scalar_grid_values():
    """Scalar analysis recovers known coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    expected = np.zeros(basis.coefficient_count)
    expected[1] = 1.0
    expected[3] = -0.25
    values = transform.synthesize_scalar(expected)

    actual = transform.analyze_scalar_samples(values, input_grid=grid)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("helmholtz", [False, True])
@pytest.mark.parametrize("remap", [False, True])
def test_sample_analysis_inherits_the_same_grid_fit_policy(helmholtz, remap):
    """Changing the sample layout does not change the weighted, regularized fit."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(
        basis, grid, sqrt_weights=np.linspace(0.3, 2.0, grid.size), reg_lambda=0.2
    )
    input_grid = SphericalGrid(theta=grid.theta, phi=grid.phi)
    field = "helmholtz" if helmholtz else "scalar"
    shape = (2, grid.size) if helmholtz else (grid.size,)
    values = np.random.default_rng(28).normal(size=(*shape, 3))
    analyze = getattr(transform, f"analyze_{field}")
    expected = np.stack([analyze(values[..., i], solver="svd") for i in range(3)], axis=-1)

    actual = getattr(transform, f"analyze_{field}_samples")(
        values,
        input_grid=input_grid,
        remap=identity_linear_map(shape) if remap else None,
        solver="svd",
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("helmholtz", [False, True])
@pytest.mark.parametrize("remap", [False, True])
def test_sample_analysis_can_override_the_fit_policy(helmholtz, remap):
    """Zero regularization and unit weights explicitly disable inherited settings."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(
        basis, grid, sqrt_weights=np.linspace(0.3, 2.0, grid.size), reg_lambda=0.2
    )
    plain = SphericalTransform(basis, grid)
    field = "helmholtz" if helmholtz else "scalar"
    shape = (2, grid.size) if helmholtz else (grid.size,)
    values = np.random.default_rng(29).normal(size=shape)
    expected = getattr(plain, f"analyze_{field}")(values, solver="svd")

    actual = getattr(transform, f"analyze_{field}_samples")(
        values,
        input_grid=grid,
        remap=identity_linear_map(shape) if remap else None,
        sqrt_weights=np.ones(grid.size),
        reg_lambda=0.0,
        solver="svd",
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    assert transform.reg_lambda == 0.2


@pytest.mark.parametrize("helmholtz", [False, True])
def test_direct_sample_analysis_keeps_weights_on_their_grid(helmholtz):
    """Another input grid inherits smoothness, but not the bound grid's weights."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    input_grid = SphericalGrid(
        theta=grid.theta, phi=grid.phi + 2.5, area_weights=np.linspace(2.0, 0.5, grid.size)
    )
    transform = SphericalTransform(
        basis,
        grid,
        sqrt_weights=np.linspace(0.3, 2.0, grid.size),
        reg_lambda=0.2,
        area_weighted=True,
    )
    reference = SphericalTransform(basis, input_grid, reg_lambda=0.2, area_weighted=True)
    field = "helmholtz" if helmholtz else "scalar"
    shape = (2, input_grid.size) if helmholtz else (input_grid.size,)
    values = np.random.default_rng(30).normal(size=shape)
    expected = getattr(reference, f"analyze_{field}")(values, solver="svd")

    actual = getattr(transform, f"analyze_{field}_samples")(
        values, input_grid=input_grid, solver="svd"
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


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

    with pytest.raises(TypeError, match="ScalarBasis"):
        transform.with_basis(object())


def test_with_basis_honors_an_explicit_evaluation_algorithm():
    """A compatible coefficient layout must not erase an algorithm choice."""
    transform = SphericalTransform(SHBasis(3, 2), _regular_grid())
    scipy_basis = SHBasis(3, 2, legendre_method="scipy")
    rebound = transform.with_basis(scipy_basis)
    assert rebound is not transform
    assert rebound.basis is scipy_basis
    assert rebound is transform.with_basis(scipy_basis)
    np.testing.assert_allclose(rebound.scalar_synthesis_array, transform.scalar_synthesis_array)


def test_explicit_empty_solver_name_is_not_treated_as_default():
    """An invalid explicit solver selection fails visibly."""
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid())

    with pytest.raises(ValueError, match="Solver must be one of"):
        transform.analyze_scalar(np.zeros(transform.grid.size), solver="")


def test_rotated_gradient_analysis_matches_dense_least_squares():
    """Structured potential analysis preserves the dense definition."""
    basis = SHBasis(3, 3, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid, area_weighted=True)
    scale = np.linspace(0.5, 1.5, basis.coefficient_count)
    synthesis = (
        np.asarray(transform.rhat_cross_gradient_array) * scale.reshape(1, 1, -1)
    ).reshape(2 * grid.size, basis.coefficient_count)
    expected = dense_full_rank_least_squares_map(
        synthesis,
        sqrt_weights=transform.helmholtz_sqrt_weights,
        input_shape=(2, grid.size),
        output_shape=(basis.coefficient_count,),
    )

    observed = transform.rhat_cross_gradient_analysis_operator(coefficient_scale=scale)

    np.testing.assert_allclose(
        observed.to_matrix(backend="numpy"),
        expected.to_matrix(backend="numpy"),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("field_type", ["scalar", "helmholtz"])
@pytest.mark.parametrize("mean_free", [False, True])
def test_structured_sh_normal_matrix_matches_explicit_system(field_type, mean_free, monkeypatch):
    """Memory-bounded SH normals preserve the explicit system."""
    basis = SHBasis(3, 3, mean_free=mean_free)
    grid = _regular_grid()
    weights = np.random.default_rng(47).uniform(
        0.3, 2.0, size=(2, grid.size) if field_type == "helmholtz" else grid.size
    )
    transform = SphericalTransform(basis, grid, reg_lambda=0.2, sqrt_weights=weights)
    problem = (
        transform.scalar_least_squares_problem
        if field_type == "scalar"
        else transform.helmholtz_least_squares_problem
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Structured normal construction must not assemble the full system.")

    monkeypatch.setattr(problem, "system_matrix", unexpected)
    observed = problem.dense_normal_matrix()
    assert isinstance(observed, get_array_module().ndarray)
    system = np.asarray(problem.system_operator.to_matrix(backend="numpy"))

    np.testing.assert_allclose(observed, system.T @ system, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("weight_kind", ["none", "shared", "component"])
def test_helmholtz_normal_blocks_preserve_the_weighted_norm(
    complex_values, weight_kind, monkeypatch
):
    """Bounded products preserve complex adjoints and unequal component weights."""
    xp = get_array_module()
    rng = np.random.default_rng(817)
    theta, phi = rng.normal(size=(2, 13, 5))
    if complex_values:
        theta = theta + 1j * rng.normal(size=theta.shape)
        phi = phi + 1j * rng.normal(size=phi.shape)
    weights = rng.uniform(0.2, 2.0, size=(2, 13))
    if weight_kind == "none":
        weights[:] = 1
    elif weight_kind == "shared":
        weights[1] = weights[0]
    synthesis = np.block([[-theta, -phi], [-phi, theta]])
    weighted = weights.reshape(-1, 1) * synthesis
    expected = weighted.T.conj() @ weighted
    # Force several blocks and a shorter final block, without large arrays.
    import kompe.math.linear_map as linear_map_module

    monkeypatch.setattr(linear_map_module, "_WEIGHTED_PRODUCT_WORK_BYTES", 160)
    inputs = (
        xp.asarray(theta),
        xp.asarray(phi),
        None if weight_kind == "none" else xp.asarray(weights),
    )
    import kompe.basis as basis_module

    build = basis_module._helmholtz_normal_matrix
    actual = build(*inputs)
    assert isinstance(actual, xp.ndarray)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
    coefficients = rng.normal(size=10) + 1j * rng.normal(size=10)
    np.testing.assert_allclose(
        np.vdot(coefficients, np.asarray(actual) @ coefficients),
        np.linalg.norm(weighted @ coefficients) ** 2,
        rtol=1e-13,
    )
    if jax_enabled():
        import jax

        with jax.checking_leaks():
            np.testing.assert_allclose(jax.jit(build)(*inputs), expected, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("field", ["scalar", "helmholtz"])
@pytest.mark.parametrize("weights", ["none", "area", "explicit"])
def test_sample_analysis_reuses_the_bound_fit(field, weights, monkeypatch):
    """Equivalent sample requests share the bound problem and its factorization."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    kwargs = {"area_weighted": weights == "area"}
    if weights == "explicit":
        kwargs["sqrt_weights"] = get_array_module().linspace(0.3, 2.0, grid.size)
    transform = SphericalTransform(basis, grid, reg_lambda=0.2, **kwargs)
    shape = (2, grid.size) if field == "helmholtz" else (grid.size,)
    values = np.random.default_rng(39).normal(size=(*shape, 3))
    analyze = getattr(transform, f"analyze_{field}")
    expected = np.stack([analyze(values[..., i], solver="normal_pinv") for i in range(3)], axis=-1)
    problem = getattr(transform, f"{field}_least_squares_problem")
    solver = LeastSquaresSolver("normal_pinv")
    solve = solver.solve
    calls = []

    def record(actual_problem, rhs):
        calls.append(actual_problem)
        return solve(actual_problem, rhs)

    monkeypatch.setattr(solver, "solve", record)

    def unexpected(*args, **kwargs):
        pytest.fail("A default same-grid fit must not hash or transfer its own weights.")

    fingerprint = spherical_transform_module.array_fingerprint
    monkeypatch.setattr(spherical_transform_module, "array_fingerprint", unexpected)
    actual = getattr(transform, f"analyze_{field}_samples")(values, input_grid=grid, solver=solver)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    # Copies with exact identities share the same fit too.
    monkeypatch.setattr(spherical_transform_module, "array_fingerprint", fingerprint)
    input_grid = SphericalGrid(theta=grid.theta, phi=grid.phi, area_weights=grid.area_weights)
    actual = getattr(transform, f"analyze_{field}_samples")(
        values,
        input_grid=input_grid,
        sqrt_weights=None if weights != "explicit" else transform.sqrt_weights.copy(),
        reg_lambda=0.2,
        solver=solver,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    assert calls == [problem, problem]
    assert not transform._analysis_transforms


@pytest.mark.parametrize("change", ["coordinates", "measure", "weights", "regularization"])
def test_sample_analysis_keeps_distinct_fit_settings_separate(change):
    """Reuse requires exact geometry, measures, weights, and evaluation settings."""
    basis = SHBasis(3, 2, mean_free=True)
    points = _regular_grid()
    grid = SphericalGrid(theta=points.theta, phi=points.phi, area_weights=np.ones(points.size))
    transform = SphericalTransform(basis, grid, reg_lambda=0.2, area_weighted=True)
    input_grid = grid
    weights, reg_lambda = None, 0.2
    if change == "coordinates":
        input_grid = SphericalGrid(
            theta=grid.theta, phi=grid.phi + 1e-10, area_weights=grid.area_weights
        )
        assert input_grid.same_as(grid)
    elif change == "measure":
        input_grid = SphericalGrid(
            theta=grid.theta, phi=grid.phi, area_weights=np.linspace(0.5, 2.0, grid.size)
        )
    elif change == "weights":
        weights = np.linspace(0.5, 2.0, grid.size)
    elif change == "regularization":
        reg_lambda = 0.1
    values = np.random.default_rng(40).normal(size=grid.size)
    reference = SphericalTransform(
        basis, input_grid, sqrt_weights=weights, reg_lambda=reg_lambda, area_weighted=True
    )

    actual = transform.analyze_scalar_samples(
        values,
        input_grid=input_grid,
        sqrt_weights=weights,
        reg_lambda=reg_lambda,
        solver="svd",
    )

    np.testing.assert_allclose(
        actual, reference.analyze_scalar(values, solver="svd"), rtol=1e-11, atol=1e-12
    )
    cached = next(iter(transform._analysis_transforms.values()))
    assert cached is not transform
    assert cached.grid.signature == input_grid.signature
    assert cached.basis is basis


def test_spherical_transform_caches_sample_analysis_transforms_by_grid():
    """The bound grid reuses this transform; other grids have separate cached fits."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    shifted_grid = SphericalGrid(lat=grid.lat, lon=grid.lon + 1.0)
    transform = SphericalTransform(basis, grid)

    transform.analyze_scalar_samples(np.zeros(grid.size), input_grid=grid)
    assert not transform._analysis_transforms
    transform.analyze_scalar_samples(np.zeros(shifted_grid.size), input_grid=shifted_grid)
    shifted_transform = next(iter(transform._analysis_transforms.values()))
    transform.analyze_scalar_samples(np.zeros(grid.size), input_grid=grid)
    transform.analyze_scalar_samples(np.zeros(shifted_grid.size), input_grid=shifted_grid)

    assert len(transform._analysis_transforms) == 1
    assert next(iter(transform._analysis_transforms.values())) is shifted_transform


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
    transform.analyze_scalar_samples(values, input_grid=first)
    transform.analyze_scalar_samples(values, input_grid=second)

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
        sqrt_weights=supplied_weights,
        reg_lambda=0.1,
    )
    cached_transform = next(iter(transform._analysis_transforms.values()))
    supplied_weights[0] = 2.0

    transform.analyze_scalar_samples(
        values,
        input_grid=grid,
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
        sqrt_weights=supplied_weights,
        reg_lambda=0.1,
    )

    assert len(transform._analysis_transforms) == 2


def test_direct_analysis_cache_treats_zero_regularization_as_none():
    """Equivalent unregularized requests reuse the bound analysis transform."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    values = np.zeros(grid.size)

    transform.analyze_scalar_samples(values, input_grid=grid, reg_lambda=0.0)
    transform.analyze_scalar_samples(values, input_grid=grid, reg_lambda=None)

    assert not transform._analysis_transforms
    assert "scalar_least_squares_problem" in transform.__dict__


# Spectral regularization


def test_spherical_transform_regularization_uses_diagonal_operators():
    """Keep surface smoothness structured in least-squares."""
    basis = SHBasis(4, 2, mean_free=True)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=1.0)
    n = np.asarray(basis.n, dtype=float)
    q = 1.0 / (2.0 * n + 1.0)
    mu = n * (n + 1.0)
    scalar_weights = np.sqrt(q * mu)
    helmholtz_weights = np.broadcast_to(np.sqrt(q) * mu, (2, basis.coefficient_count))
    helmholtz_coeffs = np.vstack(
        [
            np.linspace(0.0, 1.0, basis.coefficient_count),
            np.linspace(1.0, 2.0, basis.coefficient_count),
        ]
    )
    scalar_coeffs = np.linspace(0.0, 1.0, basis.coefficient_count)

    scalar_regularization = transform.scalar_regularization_operator
    helmholtz_regularization = transform.helmholtz_regularization_operator

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


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_transform_rejects_invalid_tolerance(tolerance):
    """Invalid solver tolerance fails at transform construction."""
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        SphericalTransform(SHBasis(3, 2), _regular_grid(), tolerance=tolerance)


def test_transform_rejects_boolean_tolerance():
    """A Boolean is not a meaningful numerical tolerance."""
    with pytest.raises(TypeError, match="finite non-negative scalar"):
        SphericalTransform(SHBasis(3, 2), _regular_grid(), tolerance=True)


def test_surface_smoothness_regularization_matches_parseval_weights():
    """Regularizer norms reproduce analytic Schmidt-basis energies."""
    basis = SHBasis(5, 3, mean_free=False)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=1.0)
    rng = np.random.default_rng(17)
    scalar = rng.normal(size=basis.coefficient_count)
    helmholtz = rng.normal(size=(2, basis.coefficient_count))
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
            2, basis.coefficient_count
        )[0],
        transform.helmholtz_regularization_operator.diagonal(backend="numpy").reshape(
            2, basis.coefficient_count
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
    values = np.asarray(basis.scalar_evaluation_array(grid))
    gradient = np.asarray(basis.surface_gradient_array(grid))
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
    gradient = np.asarray(basis.surface_gradient_array(grid))
    normalized_area = solid_angle.reshape(-1) / (4.0 * np.pi)
    gradient_norms = np.sum(normalized_area[None, :, None] * gradient**2, axis=(0, 1))

    np.testing.assert_allclose(
        gradient_norms,
        basis.scalar_smoothness_operator().diagonal() ** 2,
        rtol=1e-12,
        atol=1e-12,
    )


def test_scalar_smoothness_is_invariant_to_log_reference_mode():
    """A logarithmic reference change affects only the free mean."""
    basis = SHBasis(5, 3, mean_free=False)
    transform = SphericalTransform(basis, _regular_grid(), reg_lambda=1.0)
    coefficients = np.linspace(-1.0, 1.0, basis.coefficient_count)
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
        transform.apply_scalar_regularization(np.zeros(basis.coefficient_count))
    with pytest.raises(RuntimeError, match="Helmholtz regularization requires reg_lambda"):
        transform.apply_helmholtz_regularization(np.zeros((2, basis.coefficient_count)))


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
    expected = np.zeros((2, basis.coefficient_count))
    expected[0, 1] = 1.0
    expected[1, 3] = -0.5
    values = transform.synthesize_helmholtz(expected)

    actual = transform.analyze_helmholtz_samples(values, input_grid=grid)
    direct = transform.analyze_helmholtz(values)

    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(direct, expected, atol=1e-10)
    assert "helmholtz_analysis_operator" not in transform.__dict__


def test_spherical_transform_batches_direct_analysis():
    """Direct SH analysis handles multiple RHS columns at once."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    scalar_coeffs = np.zeros((2, basis.coefficient_count))
    scalar_coeffs[0, 1] = 1.0
    scalar_coeffs[1, 3] = -0.25
    scalar_values = np.stack([transform.synthesize_scalar(row) for row in scalar_coeffs], axis=-1)
    vector_coeffs = np.zeros((2, 2, basis.coefficient_count))
    vector_coeffs[0, 0, 1] = 1.0
    vector_coeffs[1, 1, 3] = -0.5
    vector_values = np.stack(
        [transform.synthesize_helmholtz(row) for row in vector_coeffs], axis=-1
    )

    scalar_actual = transform.analyze_scalar_samples(scalar_values, input_grid=grid)
    vector_actual = transform.analyze_helmholtz_samples(vector_values, input_grid=grid)

    np.testing.assert_allclose(scalar_actual, scalar_coeffs.T, atol=1e-10)
    np.testing.assert_allclose(vector_actual, np.moveaxis(vector_coeffs, 0, -1), atol=1e-10)


def test_spherical_transform_least_squares_use_operator_properties():
    """Least-squares setup should not force dense attributes."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)

    scalar_problem = transform.scalar_least_squares_problem
    helmholtz_problem = transform.helmholtz_least_squares_problem

    assert scalar_problem.data_operators[0] is transform.scalar_synthesis_operator
    assert helmholtz_problem.data_operators[0] is transform.helmholtz_synthesis_operator
    assert "scalar_synthesis_array" not in transform.__dict__
    assert "helmholtz_synthesis_array" not in transform.__dict__


# Cubed-sphere remapping and analysis


def test_native_cs_transform_synthesizes_from_sparse_operator_paths(monkeypatch):
    """Native CS synthesis can apply sparse operators."""
    basis = GlobalCSBasis(4)
    grid = SphericalGrid(
        theta=basis.mesh.theta, phi=basis.mesh.phi, area_weights=basis.mesh.cell_areas.reshape(-1)
    )
    transform = SphericalTransform(basis, grid)
    derivatives = basis.mesh.operators._native_derivatives
    theta = derivatives["theta"].toarray()
    phi = derivatives["phi"].toarray()

    scalar_coeffs = np.linspace(0.0, 1.0, basis.coefficient_count)
    vector_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])
    expected_helmholtz = np.stack(
        [
            -theta @ vector_coeffs[0] - phi @ vector_coeffs[1],
            -phi @ vector_coeffs[0] + theta @ vector_coeffs[1],
        ]
    )

    def fail_evaluate_on_grid(*args, **kwargs):
        raise AssertionError("native CS synthesis should use operator paths")

    monkeypatch.setattr(basis, "scalar_evaluation_array", fail_evaluate_on_grid)

    np.testing.assert_allclose(transform.synthesize_scalar(scalar_coeffs), scalar_coeffs)
    np.testing.assert_allclose(
        transform.synthesize_scalar(scalar_coeffs, gradient_component="theta"),
        theta @ scalar_coeffs,
    )
    np.testing.assert_allclose(
        transform.synthesize_scalar(scalar_coeffs, gradient_component="phi"), phi @ scalar_coeffs
    )
    np.testing.assert_allclose(transform.synthesize_helmholtz(vector_coeffs), expected_helmholtz)
    assert "scalar_synthesis_array" not in transform.__dict__
    assert "helmholtz_synthesis_array" not in transform.__dict__


def test_spherical_transform_reuses_scalar_grid_remap(monkeypatch):
    """Scalar analysis reuses a cached CS remap operator."""
    GlobalCSRemapper._shared_remap_matrix_cache.clear()
    basis = SHBasis(3, 2, mean_free=True)
    remapping_basis = GlobalCSBasis(8)
    source_basis = GlobalCSBasis(10)
    target_grid = SphericalGrid(
        theta=remapping_basis.mesh.theta,
        phi=remapping_basis.mesh.phi,
        area_weights=remapping_basis.mesh.cell_areas.reshape(-1),
    )
    input_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    values = np.column_stack(
        [np.sin(np.deg2rad(input_grid.theta)), np.cos(np.deg2rad(input_grid.phi))]
    )
    transform = SphericalTransform(basis, target_grid)
    calls = 0
    original = remapping_basis.remapper.build_scalar_grid_remap_matrix

    def counted_build_scalar_grid_remap_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        remapping_basis.remapper,
        "build_scalar_grid_remap_matrix",
        counted_build_scalar_grid_remap_matrix,
    )

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("supported CS remaps should use cached operators")

    monkeypatch.setattr(remapping_basis.remapper, "interpolate_scalar", fail_interpolate_scalar)

    projected_1 = transform.analyze_scalar_samples(
        values,
        input_grid=input_grid,
        remap=remapping_basis.remapper.scalar_operator(input_grid, target_grid),
    )
    projected_2 = transform.analyze_scalar_samples(
        values,
        input_grid=input_grid,
        remap=remapping_basis.remapper.scalar_operator(input_grid, target_grid),
    )

    assert calls == 1
    assert projected_1.shape == (basis.coefficient_count, 2)
    np.testing.assert_allclose(projected_2, projected_1)


def test_matching_grid_remap_is_identity(monkeypatch):
    """Matching grids supply an identity remap, without interpolation."""
    basis = SHBasis(3, 2, mean_free=True)
    remapping_basis = GlobalCSBasis(8)
    grid = SphericalGrid(
        theta=remapping_basis.mesh.theta,
        phi=remapping_basis.mesh.phi,
        area_weights=remapping_basis.mesh.cell_areas.reshape(-1),
    )
    values = np.column_stack([np.sin(np.deg2rad(grid.theta)), np.cos(np.deg2rad(grid.phi))])
    transform = SphericalTransform(basis, grid)

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("matching grids should not interpolate")

    monkeypatch.setattr(remapping_basis.remapper, "interpolate_scalar", fail_interpolate_scalar)

    projected = transform.analyze_scalar_samples(
        values, input_grid=grid, remap=remapping_basis.remapper.scalar_operator(grid, grid)
    )

    assert projected.shape == (basis.coefficient_count, 2)
    np.testing.assert_allclose(projected, transform.analyze_scalar(values))


def test_remapped_sample_analysis_applies_target_fit_options(monkeypatch):
    """Regularization and tolerance configure the post-remap analysis."""
    basis = SHBasis(3, 2, mean_free=True)
    remapping_basis = GlobalCSBasis(6)
    source_basis = GlobalCSBasis(8)
    input_grid = source_basis.native_grid
    values = np.sin(np.deg2rad(input_grid.theta))
    transform = SphericalTransform(basis, remapping_basis.native_grid, tolerance=1e-10)
    recorded = {}
    original = transform._sample_analysis_transform

    def record_analysis_transform(*args, **kwargs):
        recorded.update(kwargs)
        child = original(*args, **kwargs)
        recorded["tolerance"] = child.tolerance
        return child

    monkeypatch.setattr(transform, "_sample_analysis_transform", record_analysis_transform)
    projected = transform.analyze_scalar_samples(
        values,
        input_grid=input_grid,
        remap=remapping_basis.remapper.scalar_operator(input_grid, transform.grid),
        reg_lambda=1e-3,
    )

    assert projected.shape == (basis.coefficient_count,)
    assert recorded["reg_lambda"] == 1e-3
    assert recorded["tolerance"] == 1e-10


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
            remap=remapping_basis.remapper.scalar_operator(input_grid, transform.grid),
            sqrt_weights=np.ones(input_grid.size),
        )


def test_sample_analysis_checks_remap_domain_and_codomain():
    """An explicit map must connect the supplied input and fit grids."""
    grid = _regular_grid()
    transform = SphericalTransform(SHBasis(3, 2), grid)
    with pytest.raises(ValueError):
        transform.analyze_scalar_samples(
            np.ones(grid.size), input_grid=grid, remap=identity_linear_map((grid.size + 1,))
        )


def test_spherical_transform_reuses_helmholtz_grid_remap(monkeypatch):
    """Helmholtz analysis reuses a cached CS remap operator."""
    GlobalCSRemapper._shared_remap_matrix_cache.clear()
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
    values = np.stack([theta_values.T, phi_values.T])
    transform = SphericalTransform(basis, target_grid)
    calls = 0
    original = remapping_basis.remapper.build_tangential_grid_remap_matrix

    def counted_build_tangential_grid_remap_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        remapping_basis.remapper,
        "build_tangential_grid_remap_matrix",
        counted_build_tangential_grid_remap_matrix,
    )

    def fail_interpolate_vector(*args, **kwargs):
        raise AssertionError("supported CS remaps should use cached operators")

    monkeypatch.setattr(remapping_basis.remapper, "interpolate_vector", fail_interpolate_vector)

    projected_1 = transform.analyze_helmholtz_samples(
        values,
        input_grid=input_grid,
        remap=remapping_basis.remapper.tangential_operator(input_grid, target_grid),
    )
    projected_2 = transform.analyze_helmholtz_samples(
        values,
        input_grid=input_grid,
        remap=remapping_basis.remapper.tangential_operator(input_grid, target_grid),
    )

    assert calls == 1
    assert projected_1.shape == (2, basis.coefficient_count, 2)
    np.testing.assert_allclose(projected_2, projected_1)


def test_cs_scalar_remap_operator_matches_interpolation():
    """Cached scalar remap matches the legacy CS interpolation."""
    source_basis = GlobalCSBasis(8)
    target_basis = GlobalCSBasis(6)
    source_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    target_grid = SphericalGrid(theta=target_basis.mesh.theta, phi=target_basis.mesh.phi)
    values = np.sin(np.deg2rad(source_grid.theta)) + 0.25 * np.cos(np.deg2rad(source_grid.phi))

    operator = target_basis.remapper.scalar_operator(source_grid, target_grid)
    actual = operator @ values
    expected = target_basis.remapper.interpolate_scalar(
        values, source_grid.theta, source_grid.phi, target_grid.theta, target_grid.phi
    )

    assert operator is target_basis.remapper.scalar_operator(source_grid, target_grid)
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

    operator = target_basis.remapper.tangential_operator(source_grid, target_grid)
    actual = (operator @ values.reshape(-1)).reshape(2, target_grid.size)
    expected_theta, expected_phi, _ = target_basis.remapper.interpolate_vector(
        theta_component,
        phi_component,
        np.zeros_like(theta_component),
        source_grid.theta,
        source_grid.phi,
        target_grid.theta,
        target_grid.phi,
    )
    expected = np.vstack([expected_theta, expected_phi])

    assert operator is target_basis.remapper.tangential_operator(source_grid, target_grid)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_cs_tangential_remap_matrix_cache_is_shared(monkeypatch):
    """Equivalent CS remaps share sparse matrix construction."""
    GlobalCSRemapper._shared_remap_matrix_cache.clear()
    source_basis = GlobalCSBasis(8)
    target_basis = GlobalCSBasis(6)
    equivalent_target_basis = GlobalCSBasis(6)
    source_grid = SphericalGrid(theta=source_basis.mesh.theta, phi=source_basis.mesh.phi)
    target_grid = SphericalGrid(theta=target_basis.mesh.theta, phi=target_basis.mesh.phi)
    values = np.vstack(
        [np.sin(np.deg2rad(source_grid.theta)), np.cos(np.deg2rad(source_grid.phi))]
    )

    first_operator = target_basis.remapper.tangential_operator(source_grid, target_grid)

    def fail_build(*args, **kwargs):
        raise AssertionError("equivalent remap matrix should come from shared cache")

    monkeypatch.setattr(
        equivalent_target_basis.remapper, "build_tangential_grid_remap_matrix", fail_build
    )

    second_operator = equivalent_target_basis.remapper.tangential_operator(
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
    expected = basis.remapper.interpolate_scalar(
        coeffs, basis.mesh.theta, basis.mesh.phi, target.theta, target.phi
    )

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("scalar operator should use the remap LinearMap path")

    monkeypatch.setattr(basis.remapper, "interpolate_scalar", fail_interpolate_scalar)

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

    expected_gradient = np.tensordot(basis.surface_gradient_array(target), scalar_coeffs, axes=1)
    expected_rxgrad = np.tensordot(basis.rhat_cross_gradient_array(target), scalar_coeffs, axes=1)
    expected_helmholtz = np.tensordot(
        basis.helmholtz_synthesis_array(target), helmholtz_coeffs, axes=2
    )

    def fail_interpolate_vector(*args, **kwargs):
        raise AssertionError("vector operator should use the remap LinearMap path")

    monkeypatch.setattr(basis.remapper, "interpolate_vector", fail_interpolate_vector)

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
    projected_rows = transform.analyze_scalar_samples(value_rows.T, input_grid=target)

    assert coeffs.shape == (basis.coefficient_count,)
    assert projected_rows.shape == coeff_rows.T.shape
    np.testing.assert_allclose(transform.synthesize_scalar(coeffs), value_rows[0])
    for projected, expected_values in zip(projected_rows.T, value_rows, strict=True):
        np.testing.assert_allclose(transform.synthesize_scalar(projected), expected_values)


def test_direct_sample_analysis_uses_the_analysis_transform_layout():
    """A native direct analysis owns the returned batch layout."""

    basis = GlobalCSBasis(4)
    target = SphericalGrid(theta=basis.mesh.theta + 1e-3, phi=basis.mesh.phi)
    transform = SphericalTransform(basis, target)
    values = np.column_stack(
        [np.arange(basis.coefficient_count), -np.arange(basis.coefficient_count)]
    )

    projected = transform.analyze_scalar_samples(values, input_grid=basis.native_grid)

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


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
@pytest.mark.parametrize("helmholtz", [False, True])
@pytest.mark.parametrize("layout", ["single", "flat", "data_first", "multi_batch"])
def test_optimized_analysis_matches_solver_array_contract(backend, helmholtz, layout):
    """Optimizations preserve backend, component axes, and RHS ordering."""
    with backend_context(backend):
        basis = SHBasis(3, 2, mean_free=True) if helmholtz else GlobalCSBasis(4)
        grid = _regular_grid() if helmholtz else basis.native_grid
        transform = SphericalTransform(basis, grid)
        reference = SphericalTransform(basis, grid, sqrt_weights=np.ones(grid.size))
        data_shape = (2, grid.size) if helmholtz else (grid.size,)
        batch_shape = () if layout in ("single", "flat") else (3,)
        if layout == "multi_batch":
            batch_shape = (3, 4)
        values = np.random.default_rng(42).normal(size=data_shape + batch_shape)
        if layout == "flat":
            values = values.reshape(-1)
        if helmholtz:
            actual = transform.analyze_helmholtz(values)
            expected = reference.analyze_helmholtz(values, solver="svd")
            solution_shape = (2, basis.coefficient_count)
        else:
            actual = transform.analyze_scalar(values)
            expected = reference.analyze_scalar(values, solver="svd")
            solution_shape = (basis.coefficient_count,)
            assert "scalar_least_squares_problem" not in transform.__dict__

        assert isinstance(actual, get_array_module().ndarray)
        assert actual.shape == expected.shape == solution_shape + batch_shape
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("square_batch", [False, True])
def test_native_scalar_sample_analysis_preserves_batch_columns(weighted, square_batch):
    """Data axes lead even when the batch size equals the number of points."""
    basis = GlobalCSBasis(4)
    grid = basis.native_grid
    transform = SphericalTransform(
        basis, grid, sqrt_weights=np.ones(grid.size) if weighted else None
    )
    batch_size = grid.size if square_batch else 3
    values = np.arange(batch_size * grid.size, dtype=float).reshape(grid.size, batch_size)

    actual = transform.analyze_scalar_samples(values, input_grid=grid)

    assert actual.shape == values.shape
    np.testing.assert_allclose(actual, values, atol=1e-12)


@pytest.mark.parametrize("helmholtz", [False, True])
@pytest.mark.parametrize("remap", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_sample_analysis_preserves_arbitrary_trailing_batches(helmholtz, remap, batch_shape):
    """Direct and remapped fits compose with synthesis without layout changes."""
    xp = get_array_module()
    source = GlobalCSBasis(6).native_grid
    target_basis = GlobalCSBasis(4)
    grid = target_basis.native_grid if remap else source
    basis = SHBasis(3, 2, mean_free=True)
    transform = SphericalTransform(basis, grid)
    field = "helmholtz" if helmholtz else "scalar"
    shape = (2, source.size) if helmholtz else (source.size,)
    values = xp.asarray(np.random.default_rng(91).normal(size=shape + batch_shape))
    operator = None
    if remap:
        factory = (
            target_basis.remapper.tangential_operator
            if helmholtz
            else target_basis.remapper.scalar_operator
        )
        operator = factory(source, grid)
    actual = getattr(transform, f"analyze_{field}_samples")(
        values, input_grid=source, remap=operator
    )
    fit_values = values if operator is None else operator(values)
    analyze = getattr(transform, f"analyze_{field}")
    if batch_shape:
        expected = xp.stack(
            [analyze(fit_values[(..., *index)]) for index in np.ndindex(batch_shape)], axis=-1
        ).reshape(actual.shape)
    else:
        expected = analyze(fit_values)
    assert actual.shape == ((2,) if helmholtz else ()) + (basis.coefficient_count,) + batch_shape
    assert isinstance(actual, xp.ndarray)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    synthesized = getattr(transform, f"synthesize_{field}")(actual)
    assert synthesized.shape == ((2,) if helmholtz else ()) + (grid.size,) + batch_shape


def test_native_scalar_analysis_validates_shape_and_accepts_lists():
    """The identity shortcut has the same input boundary as a solved analysis."""
    basis = GlobalCSBasis(4)
    transform = SphericalTransform(basis, basis.native_grid)
    values = np.arange(basis.coefficient_count, dtype=float)
    actual = transform.analyze_scalar(values.tolist())
    assert actual.shape == values.shape
    np.testing.assert_array_equal(actual, values)
    weighted_transform = SphericalTransform(
        basis, basis.native_grid, sqrt_weights=np.ones(basis.coefficient_count)
    )
    np.testing.assert_allclose(weighted_transform.analyze_scalar(values.tolist()), actual)
    with pytest.raises(ValueError, match="incompatible with data_shape"):
        transform.analyze_scalar(values.reshape(2, -1))


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

    assert actual.shape == (2, basis.coefficient_count)
    np.testing.assert_allclose(transform.synthesize_helmholtz(actual), values, atol=1e-10)


# Configurable fits and fixed factorized inverse maps


@pytest.mark.parametrize("algorithm", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("kind", ["SH", "CS"])
def test_helmholtz_fit_algorithms_share_the_same_gauge(algorithm, kind):
    """Every algorithm recovers the same zero-mean manufactured potentials."""
    cs_basis = GlobalCSBasis(4)
    basis = SHBasis(3, 2, mean_free=False) if kind == "SH" else cs_basis
    transform = SphericalTransform(basis, cs_basis.native_grid, area_weighted=True)
    potentials = np.random.default_rng(83).normal(size=(2, basis.coefficient_count, 3))
    if kind == "SH":
        potentials[:, basis.n == 0] = 0.0
    else:
        means = np.einsum("cnt,n->ct", potentials, basis.scalar_mean_weights)
        potentials -= means[:, None, :]
    values = transform.synthesize_helmholtz(potentials)
    solver = LeastSquaresSolver(algorithm, tolerance=1e-12)

    actual = transform.analyze_helmholtz(values, solver=solver)

    np.testing.assert_allclose(actual, potentials, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        np.einsum("cnt,n->ct", actual, basis.scalar_mean_weights), 0.0, atol=1e-9
    )


@pytest.mark.parametrize("algorithm", LeastSquaresSolver.VALID_SOLVERS)
def test_restricted_cs_analysis_does_not_constrain_an_observable_mean(algorithm):
    """Removing a node removes the constant nullspace, not the field's mean."""
    from kompe.basis import BasisSubset

    parent = GlobalCSBasis(4)
    basis = BasisSubset(parent, np.arange(parent.coefficient_count - 1))
    transform = SphericalTransform(basis, parent.native_grid)
    potentials = np.ones((2, basis.coefficient_count))
    values = transform.synthesize_helmholtz(potentials)
    assert basis.omits_constant_mode()
    assert transform.helmholtz_least_squares_problem.constraints is None
    actual = transform.analyze_helmholtz(
        values, solver=LeastSquaresSolver(algorithm, tolerance=1e-12)
    )
    np.testing.assert_allclose(actual, potentials, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(transform.synthesize_helmholtz(actual), values, atol=1e-9)


@pytest.mark.parametrize("algorithm", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_helmholtz_gauges_are_independent_of_uniform_weight_scale(algorithm, scale):
    """No artificial gauge singular values enter the residual's spectrum."""
    basis = GlobalCSBasis(4)
    transform = SphericalTransform(
        basis, basis.native_grid, sqrt_weights=np.full(basis.coefficient_count, scale)
    )
    potentials = basis.project_helmholtz_mean_free(
        np.random.default_rng(3).normal(size=(2, basis.coefficient_count))
    )
    values = transform.synthesize_helmholtz(potentials)
    actual = transform.analyze_helmholtz(
        values, solver=LeastSquaresSolver(algorithm, tolerance=1e-12)
    )
    np.testing.assert_allclose(actual, potentials, rtol=1e-8, atol=1e-8)


def test_native_cs_normal_solve_reuses_its_sparse_inverse(monkeypatch):
    """The selected direct solve needs neither dense synthesis nor gauge rows."""
    basis = GlobalCSBasis(4)
    transform = SphericalTransform(basis, basis.native_grid)
    inverse = transform.helmholtz_analysis_operator
    values = np.random.default_rng(3).normal(size=(2, basis.coefficient_count, 3))

    def unexpected(*args, **kwargs):
        raise AssertionError("The native direct solve must reuse its sparse inverse.")

    monkeypatch.setattr(LeastSquaresSolver, "solve", unexpected)
    for _ in range(2):
        actual = transform.analyze_helmholtz(values, solver="normal_solve")
        np.testing.assert_allclose(actual, inverse(values), rtol=1e-12, atol=1e-12)
    assert "helmholtz_least_squares_problem" not in transform.__dict__
    assert not transform.helmholtz_synthesis_operator._dense_cache


def test_clearing_transform_cache_releases_constrained_coordinates():
    """The gauge basis and its restricted fit share the transform lifecycle."""
    basis = GlobalCSBasis(4)
    transform = SphericalTransform(basis, basis.native_grid)
    problem = transform.helmholtz_least_squares_problem
    coordinates = problem.solution_basis
    transform.clear_cache()
    assert "helmholtz_least_squares_problem" not in transform.__dict__
    assert transform.helmholtz_least_squares_problem.solution_basis is not coordinates
    assert transform.helmholtz_least_squares_problem is not problem


@pytest.mark.parametrize("helmholtz", [False, True])
@pytest.mark.parametrize("remap", [False, True])
def test_sample_fits_honor_the_shared_solver(monkeypatch, helmholtz, remap):
    """An injected solver survives direct-analysis and CS-remapping routes."""
    cs_basis = GlobalCSBasis(4)
    basis = SHBasis(3, 2)
    grid = GlobalCSBasis(6).native_grid
    transform = SphericalTransform(basis, cs_basis.native_grid)
    shape = (2, grid.size) if helmholtz else (grid.size,)
    values = np.random.default_rng(7).normal(size=(*shape, 3))
    solver = LeastSquaresSolver("svd", tolerance=1e-8)
    solve = solver.solve
    calls = []

    def record(problem, rhs):
        calls.append(problem)
        return solve(problem, rhs)

    monkeypatch.setattr(solver, "solve", record)
    analyze = (
        transform.analyze_helmholtz_samples if helmholtz else transform.analyze_scalar_samples
    )
    operator = None
    if remap:
        factory = (
            cs_basis.remapper.tangential_operator
            if helmholtz
            else cs_basis.remapper.scalar_operator
        )
        operator = factory(grid, transform.grid)
    actual = analyze(values, input_grid=grid, remap=operator, solver=solver)
    assert len(calls) == 1
    expected_shape = (2, basis.coefficient_count, 3) if helmholtz else (basis.coefficient_count, 3)
    assert actual.shape == expected_shape


@pytest.mark.parametrize("algorithm", LeastSquaresSolver.VALID_SOLVERS)
def test_constant_gauges_do_not_change_smoothness_regularization(algorithm):
    """Including SH constant modes changes gauges, not the field-fit objective."""
    full_basis = SHBasis(3, 2, mean_free=False)
    grid = GlobalCSBasis(6).native_grid
    full = SphericalTransform(full_basis, grid, reg_lambda=0.01)
    mean_free = SphericalTransform(full_basis.with_mean_free(True), grid, reg_lambda=0.01)
    values = np.random.default_rng(96).normal(size=(2, grid.size))
    solver = LeastSquaresSolver(algorithm, tolerance=1e-12)
    actual = full.analyze_helmholtz(values, solver=solver)
    expected = mean_free.analyze_helmholtz(values, solver=solver)

    np.testing.assert_allclose(actual[:, full_basis.n != 0], expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual[:, full_basis.n == 0], 0.0, atol=1e-12)


def test_default_helmholtz_fit_honors_the_environment(monkeypatch):
    """A native CS inverse must not silently replace the requested fit algorithm."""
    monkeypatch.setenv("KOMPE_LEAST_SQUARES_SOLVER", "svd")
    basis = GlobalCSBasis(4)
    transform = SphericalTransform(basis, basis.native_grid, tolerance=1e-9)
    calls = []
    solve = LeastSquaresSolver.solve

    def record(self, problem, rhs):
        calls.append((self.method, self.tolerance))
        return solve(self, problem, rhs)

    monkeypatch.setattr(LeastSquaresSolver, "solve", record)
    transform.analyze_helmholtz(np.ones((2, basis.coefficient_count)))
    assert calls == [("svd", 1e-9)]
    assert "_optimized_helmholtz_analysis_operator" not in transform.__dict__


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
    actual = operator(values)
    expected = reference.analyze_helmholtz(values, solver="normal_pinv")

    assert np not in operator._dense_cache
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    if jax_enabled():
        import jax
        import jax.numpy as jnp

        assert "jax" in type(actual).__module__
        compiled = jax.jit(operator)(jnp.asarray(values))
        np.testing.assert_allclose(compiled, expected, rtol=2e-12, atol=2e-12)


def test_full_mean_sh_helmholtz_analysis_retains_rank_deficient_fallback():
    """Constant SH gauges continue through pseudoinverse analysis."""
    basis = SHBasis(3, 2, mean_free=False)
    cs_basis = GlobalCSBasis(6)
    grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
    transform = SphericalTransform(basis, grid)

    assert transform._optimized_helmholtz_analysis_operator is None
    assert transform.helmholtz_analysis_operator.shape == (
        2 * basis.coefficient_count,
        2 * grid.size,
    )


def test_rank_deficient_helmholtz_analysis_uses_transform_tolerance(monkeypatch):
    """The configured tolerance reaches the pseudoinverse fallback."""
    recorded = {}
    weighted_tensor_pinv = spherical_transform_module.weighted_tensor_pinv

    def record_tolerance(*args, **kwargs):
        recorded["rtol"] = kwargs["rtol"]
        return weighted_tensor_pinv(*args, **kwargs)

    monkeypatch.setattr(spherical_transform_module, "weighted_tensor_pinv", record_tolerance)
    basis = SHBasis(3, 2, mean_free=False)
    cs_basis = GlobalCSBasis(6)
    grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
    transform = SphericalTransform(basis, grid, tolerance=1e-8)

    _ = transform.helmholtz_analysis_operator

    assert recorded["rtol"] == 1e-8


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
    for weights in (np.ones(grid.size - 1), np.full(grid.size, -1.0), np.full(grid.size, np.nan)):
        with pytest.raises(ValueError, match="sqrt_weights"):
            SphericalTransform(basis, grid, sqrt_weights=weights)


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
    actual = operator.matvec(values).reshape(2, basis.coefficient_count)
    assert "helmholtz_least_squares_problem" not in transform.__dict__
    api_actual = transform.analyze_helmholtz(values)
    expected = reference_transform.analyze_helmholtz(values, solver="normal_pinv")
    expected = basis.project_helmholtz_mean_free(expected)

    assert np not in operator._dense_cache
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(api_actual, actual)
    np.testing.assert_allclose(basis.scalar_mean(actual.T), np.zeros(2), atol=2e-14)
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
    batch_actual = transform.analyze_helmholtz(np.moveaxis(value_batch, 0, -1))
    batch_expected = np.stack(
        [
            basis.project_helmholtz_mean_free(
                reference_transform.analyze_helmholtz(row, solver="normal_pinv")
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

    GlobalCSRemapper.clear_shared_cache()
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
        remapping_basis.remapper.scalar_operator(
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
        scalar_coeffs = np.linspace(0.0, 1.0, basis.coefficient_count)
        scalar_values = transform.synthesize_scalar(scalar_coeffs)
        assert "jax" in type(scalar_values).__module__
        backend_dtype = to_numpy(scalar_values).dtype
        assert np.issubdtype(backend_dtype, np.floating)
        assert to_numpy(transform.scalar_synthesis_array).dtype == backend_dtype
        np.testing.assert_allclose(
            to_numpy(scalar_values), to_numpy(transform.scalar_synthesis_array) @ scalar_coeffs
        )

        vector_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])
        vector_values = transform.synthesize_helmholtz(vector_coeffs)
        assert "jax" in type(vector_values).__module__
        assert to_numpy(vector_values).dtype == backend_dtype
        assert to_numpy(transform.helmholtz_synthesis_array).dtype == backend_dtype
        np.testing.assert_allclose(
            to_numpy(vector_values),
            np.tensordot(to_numpy(transform.helmholtz_synthesis_array), vector_coeffs, 2),
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
        coeffs = jnp.linspace(0.0, 1.0, basis.coefficient_count)

        values = transform.synthesize_scalar(coeffs)

        assert "jax" in type(values).__module__
        np.testing.assert_allclose(
            to_numpy(values), transform.scalar_synthesis_array @ to_numpy(coeffs)
        )
    finally:
        set_backend(previous_backend)
