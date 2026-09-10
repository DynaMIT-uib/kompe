"""Operator-only basis evaluation, coefficient restrictions, and derivative ranges."""

import numpy as np
import pytest

from kompe import GlobalCSBasis, SECSBasis, SHBasis, SphericalGrid, SphericalTransform
from kompe.basis import BasisSubset, ScalarBasis
from kompe.cache import PersistentArrayCache
from kompe.coefficients import CoefficientSpace
from kompe.math import LinearMap, get_array_module


def test_scalar_basis_can_define_only_a_matrix_free_evaluation_operator():
    """A scalar expansion need not supply an explicit evaluation array."""

    class CosineBasis(ScalarBasis):
        kind = "cosine"
        index_names = ("degree",)
        coefficient_count = 2
        index_arrays = (np.array([0, 1]),)
        coefficient_space_signature = ("cosine",)

        def scalar_evaluation_operator(self, grid, gradient_component=None, *, persist=True):
            xp = get_array_module()
            cosine = xp.cos(xp.deg2rad(grid.theta))
            return LinearMap(
                shape=(grid.size, 2),
                dtype=cosine.dtype,
                matvec=lambda c: c[0] + c[1] * cosine,
                rmatvec=lambda f: xp.stack([xp.sum(f), xp.sum(cosine * f)]),
                backend_operands=(cosine,),
            )

    basis = CosineBasis()
    grid = SphericalGrid(theta=[10, 45, 90, 135, 170], phi=0)
    xp = get_array_module()
    values = basis.scalar_evaluation_operator(grid)(xp.asarray([2.0, 3.0]))
    expected = 2 + 3 * np.cos(np.deg2rad(grid.theta))
    np.testing.assert_allclose(values, expected)
    np.testing.assert_allclose(
        basis.scalar_evaluation_array(grid) @ xp.asarray([2.0, 3.0]), expected
    )
    transform = SphericalTransform(basis, grid, tolerance=1e-12)
    np.testing.assert_allclose(transform.analyze_scalar(values, solver="cgls"), [2, 3], atol=1e-12)
    assert transform.scalar_synthesis_operator.materialized_matrix is None


@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize(
    "name",
    [
        "scalar_synthesis",
        "gradient_theta",
        "gradient_phi",
        "surface_gradient",
        "rhat_cross_gradient",
        "helmholtz_synthesis",
    ],
)
@pytest.mark.parametrize("materialize_first", [False, True])
def test_transform_can_disable_all_persistent_evaluations(
    tmp_path, monkeypatch, subset, name, materialize_first
):
    """Scalar and vector maps honor the same policy, including after eviction."""
    cache = PersistentArrayCache(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("A transient evaluation must not access the persistent cache.")

    monkeypatch.setattr(cache, "get_or_create", unexpected)
    parent = SHBasis(3, 2, operator_cache=cache)
    basis = BasisSubset(parent, [1, 3, 5]) if subset else parent
    grid = SphericalGrid(theta=[15, 40, 110, 170], phi=[20, 80, 140, 280])
    transform = SphericalTransform(basis, grid, use_persistent_evaluation_cache=False)
    if materialize_first:
        array = getattr(transform, name + "_array")
    operator = getattr(transform, name + "_operator")
    parent.clear_cache()
    # Materialization uses retained operands, not another basis/cache lookup.
    actual = operator.to_array()
    if materialize_first:
        np.testing.assert_array_equal(actual, array)
    reference_parent = SHBasis(3, 2)
    reference_basis = BasisSubset(reference_parent, [1, 3, 5]) if subset else reference_parent
    expected = getattr(SphericalTransform(reference_basis, grid), name + "_array")
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert not tuple(tmp_path.iterdir())


def test_sh_evaluation_memory_cache_does_not_depend_on_persistence(tmp_path, monkeypatch):
    """Persistence is an I/O policy, not part of the mathematical cache key."""
    cache = PersistentArrayCache(tmp_path)
    basis = SHBasis(3, 3, operator_cache=cache)
    grid = SphericalGrid(theta=[30, 70, 120], phi=[10, 90, 200])
    operator = basis.helmholtz_synthesis_operator(grid)
    assert len(tuple((tmp_path / "sh_evaluation").glob("*.npy"))) == 2

    def unexpected(*args, **kwargs):
        pytest.fail("Existing derivative data must be reused without disk access.")

    monkeypatch.setattr(cache, "get_or_create", unexpected)
    assert basis.helmholtz_synthesis_operator(grid, persist=False) is operator
    basis.clear_cache()
    operator.to_array()


@pytest.mark.parametrize("weighted", [False, True])
def test_sh_vector_operators_share_derivatives_and_keep_cheap_normal_diagonals(
    monkeypatch, weighted
):
    """Block actions retain two derivative arrays, not stacked/rotated copies."""
    import kompe.math.linear_map as linear_map_module

    xp = get_array_module()
    basis = SHBasis(4, 3)
    grid = SphericalGrid(theta=[15, 35, 60, 120, 165], phi=[20, 80, 140, 200, 280])
    gradient = basis.surface_gradient_operator(grid)
    rotated = basis.rhat_cross_gradient_operator(grid)
    helmholtz = basis.helmholtz_synthesis_operator(grid)
    arrays = next(iter(basis._grid_cache.values()))["arrays"]
    assert set(arrays) == {("scalar_evaluation", "theta"), ("scalar_evaluation", "phi")}
    theta, phi = (
        np.asarray(arrays[("scalar_evaluation", component)]) for component in ("theta", "phi")
    )
    references = [
        np.vstack([theta, phi]),
        np.vstack([-phi, theta]),
        np.block([[-theta, -phi], [-phi, theta]]),
    ]

    def unexpected(*args, **kwargs):
        pytest.fail("SH normal diagonals must not probe coefficient columns.")

    monkeypatch.setattr(linear_map_module, "_normal_matrix_diag_from_matmat", unexpected)
    rng = np.random.default_rng(921)
    weights = np.linspace(0.5, 2, 2 * grid.size) if weighted else None
    for operator, reference in zip((gradient, rotated, helmholtz), references, strict=True):
        c = xp.asarray(
            rng.normal(size=operator.input_shape + (2, 3))
            + 1j * rng.normal(size=operator.input_shape + (2, 3))
        )
        f = xp.asarray(
            rng.normal(size=operator.output_shape + (2, 3))
            + 1j * rng.normal(size=operator.output_shape + (2, 3))
        )
        actual = operator(c)
        adjoint = operator.adjoint()(f)
        single_c = np.asarray(c[..., 0, 0]).reshape(-1).tolist()
        single_f = np.asarray(f[..., 0, 0]).reshape(-1).tolist()
        np.testing.assert_allclose(operator.matvec(single_c), reference @ single_c, atol=2e-13)
        np.testing.assert_allclose(operator.rmatvec(single_f), reference.T @ single_f, atol=2e-13)
        np.testing.assert_allclose(
            actual.reshape(operator.shape[0], -1),
            reference @ np.asarray(c).reshape(operator.shape[1], -1),
            atol=2e-13,
        )
        np.testing.assert_allclose(
            adjoint.reshape(operator.shape[1], -1),
            reference.T @ np.asarray(f).reshape(operator.shape[0], -1),
            atol=2e-13,
        )
        np.testing.assert_allclose(np.vdot(actual, f), np.vdot(c, adjoint), atol=5e-13)
        weighted_reference = reference if weights is None else weights[:, None] * reference
        np.testing.assert_allclose(
            operator.normal_matrix_diag(row_scale=weights),
            np.sum(weighted_reference**2, axis=0),
            atol=2e-13,
        )
        assert operator.materialized_matrix is None
        matrix = operator.to_matrix()
        np.testing.assert_allclose(matrix, reference, atol=2e-13)
        np.testing.assert_allclose(operator(c), actual, atol=2e-13)
        assert operator.to_matrix() is matrix
        host_matrix = operator.to_matrix(backend="numpy")
        assert isinstance(host_matrix, np.ndarray)
        np.testing.assert_allclose(host_matrix, reference, atol=2e-13)


def test_empty_sh_helmholtz_space_is_a_zero_map():
    """Omitting the sole constant mode leaves no coefficients, not an invalid shape."""
    xp = get_array_module()
    basis = SHBasis(0, 0)
    grid = SphericalGrid(theta=[30, 60, 90], phi=[0, 50, 100])
    operator = basis.helmholtz_synthesis_operator(grid)
    np.testing.assert_array_equal(operator(xp.empty((2, 0))), np.zeros((2, 3)))
    assert operator.adjoint()(xp.ones((2, 3))).shape == (2, 0)
    assert operator.to_array().shape == (2, 3, 2, 0)


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
@pytest.mark.parametrize("weighted", [False, True])
def test_scalar_transform_supports_secs_potentials(current_type, weighted):
    """Scalar fitting needs no coefficient-space Laplacian or Helmholtz gauge."""
    xp = get_array_module()
    poles = SphericalGrid(theta=[20, 60, 100], phi=[0, 70, 160])
    basis = SECSBasis(poles, current_type=current_type)
    grid = SphericalGrid(theta=np.linspace(10, 170, 41), phi=np.linspace(5, 355, 41))
    weights = np.linspace(0.5, 1.5, grid.size) if weighted else None
    transform = SphericalTransform(basis, grid, sqrt_weights=weights, tolerance=1e-12)
    coeffs = xp.asarray(np.random.default_rng(312).normal(size=(3, 2, 3)))
    values = transform.synthesize_scalar(coeffs)
    noisy = values + xp.sin(xp.arange(values.size)).reshape(values.shape) * 0.01
    matrix = np.asarray(basis.scalar_evaluation_array(grid))
    w = np.ones(grid.size) if weights is None else weights
    expected = np.linalg.lstsq(
        w[:, None] * matrix, w[:, None] * np.asarray(noisy).reshape(grid.size, -1), rcond=1e-12
    )[0]
    np.testing.assert_allclose(
        transform.analyze_scalar(noisy, solver="svd"), expected.reshape(coeffs.shape), atol=1e-12
    )
    np.testing.assert_allclose(transform.analyze_scalar(values, solver="svd"), coeffs, atol=1e-12)
    rebound = SphericalTransform(SHBasis(3, 3), grid).with_basis(basis)
    np.testing.assert_allclose(rebound.synthesize_scalar(coeffs), values)
    input_grid = SphericalGrid(theta=grid.theta + 0.2, phi=grid.phi)
    samples = basis.scalar_evaluation_operator(input_grid)(coeffs)
    np.testing.assert_allclose(
        transform.analyze_scalar_samples(samples, input_grid=input_grid, solver="svd"),
        coeffs,
        atol=1e-12,
    )
    for component in ("theta", "phi"):
        np.testing.assert_allclose(
            transform.synthesize_scalar(coeffs, component),
            basis.scalar_evaluation_operator(grid, component)(coeffs),
        )
    for name in ("helmholtz_synthesis_operator", "helmholtz_analysis_operator"):
        with pytest.raises(TypeError, match="SurfaceDifferentialBasis"):
            getattr(transform, name)
    regularized = SphericalTransform(basis, grid, reg_lambda=0.1)
    with pytest.raises(TypeError, match="surface-smoothness"):
        regularized.analyze_scalar(values)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "operation", ["scalar", "gradient", "rotated_gradient", "helmholtz", "laplacian"]
)
def test_cs_subset_operators_keep_the_parent_matrix_free(monkeypatch, native, operation):
    """Restrict the input coefficients, not the full spatial derivative."""
    parent = GlobalCSBasis(4)
    grid = parent.native_grid if native else SphericalGrid(theta=[35, 70, 120], phi=[20, 115, 250])
    indices = np.array([1, 5, 12])
    basis = BasisSubset(parent, indices)
    method = {
        "scalar": "scalar_evaluation_operator",
        "gradient": "surface_gradient_operator",
        "rotated_gradient": "rhat_cross_gradient_operator",
        "helmholtz": "helmholtz_synthesis_operator",
        "laplacian": "laplacian_evaluation_operator",
    }[operation]

    def unexpected(*args, **kwargs):
        raise AssertionError("A coefficient restriction must not materialize a dense map.")

    monkeypatch.setattr(LinearMap, "_dense_array", unexpected)
    operator = getattr(basis, method)(grid)
    full_operator = getattr(parent, method)(grid)
    shape = (2, indices.size, 3) if operation == "helmholtz" else (indices.size, 3)
    values = np.random.default_rng(418).normal(size=shape)
    full_shape = (
        (2, parent.coefficient_count, 3)
        if operation == "helmholtz"
        else (parent.coefficient_count, 3)
    )
    full_values = np.zeros(full_shape)
    full_values[..., indices, :] = values
    xp = get_array_module()
    actual = operator(xp.asarray(values))
    np.testing.assert_allclose(actual, full_operator(xp.asarray(full_values)), atol=2e-13)
    assert isinstance(actual, xp.ndarray)
    assert operator.materialized_matrix is None
    assert full_operator.materialized_matrix is None


@pytest.mark.parametrize("persistent", [False, True])
def test_subset_transform_does_not_require_a_parent_array(monkeypatch, persistent):
    """Persistent-cache policy must not change a native nodal identity into a matrix."""
    parent = GlobalCSBasis(4)
    basis = BasisSubset(parent, [1, 5, 12])
    transform = SphericalTransform(
        basis, parent.native_grid, use_persistent_evaluation_cache=persistent
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Scalar synthesis must not materialize the parent identity.")

    monkeypatch.setattr(LinearMap, "_dense_array", unexpected)
    actual = transform.synthesize_scalar(get_array_module().asarray([1.0, 2.0, 3.0]))
    expected = np.zeros(parent.coefficient_count)
    expected[[1, 5, 12]] = [1, 2, 3]
    np.testing.assert_array_equal(actual, expected)


def test_sh_subset_keeps_analytic_derivatives_and_the_diagonal_laplacian():
    """The gradient of cos(theta) is evaluated directly, not fitted to scalar modes."""
    parent = SHBasis(3, 3, mean_free=False)
    indices = np.flatnonzero((parent.n == 1) & (parent.m == 0))
    basis = BasisSubset(parent, indices)
    grid = SphericalGrid(theta=[0, 10, 60, 90, 170, 180], phi=[20, 35, 130, 205, 260, 300])
    xp = get_array_module()
    coefficients = xp.ones(1)
    gradient = basis.surface_gradient_operator(grid)(coefficients)
    np.testing.assert_allclose(gradient[0], -np.sin(np.deg2rad(grid.theta)), atol=2e-15)
    np.testing.assert_allclose(gradient[1], 0.0, atol=2e-15)
    radius = 2.3
    laplacian = basis.surface_laplacian_operator(radius)
    assert laplacian.is_diagonal
    np.testing.assert_allclose(
        basis.laplacian_evaluation_operator(grid, radius)(coefficients),
        -2 * np.cos(np.deg2rad(grid.theta)) / radius**2,
        atol=2e-15,
    )


@pytest.mark.parametrize("representation", ["scalar", "helmholtz"])
@pytest.mark.parametrize("nodal", [False, True])
def test_batched_analysis_composes_directly_with_coefficient_spaces(representation, nodal):
    """Analysis, coefficient normalization, and synthesis share scientific axes."""
    cs = GlobalCSBasis(4)
    basis = cs if nodal else SHBasis(3, 3, mean_free=False)
    space = CoefficientSpace(basis, representation=representation, mean_free=True)
    xp = get_array_module()
    original = space.project_mean_free(
        xp.asarray(np.random.default_rng(416).normal(size=space.shape + (3, 2)))
    )
    transform = SphericalTransform(basis, cs.native_grid)
    synthesize = getattr(transform, f"synthesize_{representation}")
    analyze = getattr(transform, f"analyze_{representation}")
    samples = synthesize(original)
    analyzed = analyze(samples, solver="normal_solve")
    coefficients = space.project_mean_free(analyzed)
    assert coefficients.shape == space.shape + (3, 2)
    np.testing.assert_allclose(synthesize(coefficients), samples, atol=2e-11)
    np.testing.assert_allclose(coefficients, original, atol=2e-11)
