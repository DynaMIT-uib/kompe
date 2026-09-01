"""Tests for basis interface enforcement."""

import numpy as np
import pytest

import kompe
from kompe import (
    GlobalCSBasis,
    ScalarBasis,
    SHBasis,
    SolidHarmonicOperators,
    SphericalGrid,
    SurfaceDifferentialBasis,
)
from kompe.basis import BasisSubset
from kompe.math import (
    get_backend,
    is_identity_linear_map,
    jax_enabled,
    set_backend,
    to_numpy,
)
from kompe.math.pseudoinverse import weighted_tensor_pinv
from kompe.spherical_transform import (
    SphericalTransform,
    grid_sqrt_area_weights,
    resolve_sqrt_weights,
)

# Public basis roles


def test_public_sphere_package_is_canonical():
    """Spherical basis types are available from the public package."""
    from kompe.cubed_sphere.global_basis import GlobalCSBasis as ConcreteCSBasis
    from kompe.spherical_harmonics.sh_basis import SHBasis as ConcreteSHBasis

    assert GlobalCSBasis is ConcreteCSBasis
    assert SHBasis is ConcreteSHBasis
    assert ScalarBasis is kompe.ScalarBasis
    assert "BasisSubset" not in kompe.__all__
    assert not hasattr(kompe, "BasisSubset")
    assert not hasattr(kompe, "SphericalBasis")
    assert not hasattr(kompe, "SphericalRepresentation")
    assert isinstance(kompe.__version__, str)


def test_concrete_bases_implement_basis_interface():
    """Concrete basis classes satisfy the shared metadata interface."""
    sh_basis = SHBasis(3, 3)
    cs_basis = GlobalCSBasis(4)

    assert isinstance(sh_basis, ScalarBasis)
    assert isinstance(sh_basis, SurfaceDifferentialBasis)
    assert isinstance(cs_basis, ScalarBasis)
    assert isinstance(cs_basis, SurfaceDifferentialBasis)
    assert is_identity_linear_map(cs_basis.scalar_evaluation_operator(cs_basis.native_grid))
    assert not is_identity_linear_map(sh_basis.scalar_evaluation_operator(cs_basis.native_grid))
    assert sh_basis.kind == "SH"
    assert cs_basis.kind == "CS"
    assert cs_basis.index_length == cs_basis.mesh.theta.size
    sh_basis.validate_metadata()
    cs_basis.validate_metadata()


def test_grids_are_separate_from_coefficient_bases():
    """A sample grid does not pretend to own coefficient-space metadata."""
    grid = SphericalGrid(theta=np.array([90.0]), phi=np.array([0.0]))

    assert not isinstance(grid, ScalarBasis)
    assert isinstance(SHBasis(1, 1), ScalarBasis)
    assert not hasattr(grid, "index_names")
    assert not hasattr(grid, "index_length")
    assert not hasattr(grid, "coefficient_space_signature")
    assert not hasattr(grid, "coefficients_are_compatible_with")


def test_solid_harmonics_are_separate_from_surface_bases():
    """Solid-harmonic physics wraps rather than extends an SH basis."""
    sh_basis = SHBasis(3, 2)
    cs_basis = GlobalCSBasis(4)
    solid_harmonics = SolidHarmonicOperators(sh_basis)

    assert solid_harmonics.basis is sh_basis
    assert isinstance(sh_basis, SurfaceDifferentialBasis)
    assert isinstance(cs_basis, SurfaceDifferentialBasis)
    with pytest.raises(TypeError, match="SH surface basis"):
        SolidHarmonicOperators(cs_basis)


def test_public_basis_constructors_reject_invalid_resolution():
    """Basis primitives enforce their own representation invariants."""
    with pytest.raises(TypeError, match="required positional argument"):
        GlobalCSBasis()
    with pytest.raises(ValueError, match="positive"):
        GlobalCSBasis(0)
    with pytest.raises(ValueError, match="between zero and max_degree"):
        SHBasis(2, 3)
    with pytest.raises(TypeError, match="max_degree must be an integer"):
        SHBasis(2.5, 2)


def test_basis_metadata_arrays_are_immutable_values():
    """Basis cache identity cannot drift through array mutation."""
    sh_basis = SHBasis(3, 2)
    cs_basis = GlobalCSBasis(4)

    assert sh_basis.index_names == ("n", "m")
    assert cs_basis.index_names == ("theta", "phi")

    for values in (
        sh_basis.n,
        sh_basis.m,
        sh_basis.schmidt_factors,
        sh_basis.cosine_degree,
        sh_basis.sine_order,
        cs_basis.mesh.theta,
        cs_basis.mesh.phi,
        cs_basis.mesh.cell_areas.reshape(-1),
        cs_basis.mesh.metric_tensor,
    ):
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = 0


# Spherical sample-grid values


def test_grid_equality_tolerates_roundoff_without_weakening_cache_identity():
    """Grid equality tolerates roundoff while cache signatures remain exact."""
    lat = np.array([60.0, 61.0, 62.0])
    lon = np.array([10.0, 11.0, 12.0])
    first = SphericalGrid(lat=lat, lon=lon)
    second = SphericalGrid(theta=90.0 - lat + 1e-10, phi=lon - 1e-10)
    different = SphericalGrid(lat=lat, lon=lon + np.array([0.0, 0.0, 1e-3]))

    assert first.signature != second.signature
    assert first.same_as(second)
    assert first == second
    assert not first.same_as(different)


def test_grid_retains_broadcast_shape_for_evaluated_arrays():
    """Flat numerical storage keeps the caller's useful plotting shape."""
    latitude = np.array([[60.0], [70.0]])
    longitude = np.array([[0.0, 30.0, 60.0]])

    grid = SphericalGrid(lat=latitude, lon=longitude)

    assert grid.shape == (2, 3)
    assert grid.size == 6
    np.testing.assert_allclose(grid.lat.reshape(grid.shape), np.broadcast_to(latitude, grid.shape))
    np.testing.assert_allclose(
        grid.lon.reshape(grid.shape), np.broadcast_to(longitude, grid.shape)
    )


def test_grid_owns_immutable_unambiguous_coordinates():
    """SphericalGrid identity cannot diverge from mutable caller coordinates."""
    lat = np.array([60.0, 61.0])
    grid = SphericalGrid(lat=lat, lon=[10.0, 11.0])
    lat[0] = 0.0

    np.testing.assert_allclose(grid.lat, [60.0, 61.0])
    with pytest.raises(ValueError, match="read-only"):
        grid.lat[0] = 0.0
    with pytest.raises(ValueError, match="exactly one"):
        SphericalGrid(lat=[60.0], theta=[30.0], lon=[0.0])
    with pytest.raises(ValueError, match="between -90 and 90"):
        SphericalGrid(lat=[91.0], lon=[0.0])


def test_grid_rejects_invalid_area_weights():
    """Area weights must define a finite non-negative measure."""
    with pytest.raises(ValueError, match="non-negative"):
        SphericalGrid(lat=[60.0], lon=[0.0], area_weights=[-1.0])


def test_csbasis_native_grid_comparison_tolerates_coordinate_roundoff():
    """Native CS grid matching tolerates insignificant coordinate roundoff."""
    cs_basis = GlobalCSBasis(4)
    grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)

    assert cs_basis._is_native_grid(grid)
    grid_like = type(
        "GridLike", (), {"theta": cs_basis.mesh.theta + 1e-10, "phi": cs_basis.mesh.phi - 1e-10}
    )()
    assert cs_basis._is_native_grid(grid_like)


def test_basis_coefficient_compatibility_uses_coefficient_space():
    """Compatibility depends on coefficient layout."""
    sh_basis = SHBasis(3, 2)

    assert sh_basis.coefficients_are_compatible_with(SHBasis(3, 2))
    assert sh_basis.coefficients_are_compatible_with(SHBasis(3, 2, legendre_method="scipy"))
    assert not sh_basis.coefficients_are_compatible_with(SHBasis(3, 2, min_degree=0))
    assert not sh_basis.coefficients_are_compatible_with(SHBasis(4, 2))
    assert GlobalCSBasis(4).coefficients_are_compatible_with(GlobalCSBasis(4))
    assert not GlobalCSBasis(4).coefficients_are_compatible_with(GlobalCSBasis(6))
    assert not sh_basis.coefficients_are_compatible_with(GlobalCSBasis(4))


# Shared surface operators and solid harmonics


def test_basis_identity_separates_coefficients_from_evaluation_algorithm():
    """Numerical algorithms may differ while coefficients remain compatible."""
    internal = SHBasis(3, 2)
    scipy = SHBasis(3, 2, legendre_method="scipy")
    assert internal.coefficient_space_signature == scipy.coefficient_space_signature
    assert internal.signature != scipy.signature
    assert internal.signature == SHBasis(3, 2).signature
    assert GlobalCSBasis(4).signature != GlobalCSBasis(6).signature
    assert internal.root_basis is internal
    subset = BasisSubset(internal, coefficient_indices=[0, 1])
    assert subset.root_basis is internal
    assert BasisSubset(subset, coefficient_indices=[0]).root_basis is internal


def test_custom_bases_must_define_coefficient_identity():
    """Matching index names and sizes alone do not identify a basis."""

    class MissingIdentity(ScalarBasis):
        def scalar_evaluation_array(self, grid, derivative=None):
            return np.ones((grid.size, 1))

    with pytest.raises(TypeError, match="coefficient_space_signature"):
        MissingIdentity()


def test_surface_operator_builders_match_component_arrays():
    """Surface operators assemble the expected component arrays."""
    cs_basis = GlobalCSBasis(8)
    grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)

    G = cs_basis.scalar_evaluation_array(grid)
    G_theta = cs_basis.scalar_evaluation_array(grid, derivative="theta")
    G_phi = cs_basis.scalar_evaluation_array(grid, derivative="phi")
    gradient = cs_basis.surface_gradient_array(grid)
    rotated = cs_basis.rhat_cross_gradient_array(grid)
    helmholtz = cs_basis.helmholtz_synthesis_array(grid)
    laplacian = cs_basis.surface_laplacian_operator()
    laplacian_matrix = cs_basis.surface_laplacian_operator().to_matrix()

    np.testing.assert_allclose(G, cs_basis.scalar_evaluation_array(grid))
    np.testing.assert_allclose(gradient, np.array([G_theta, G_phi]))
    np.testing.assert_allclose(rotated, np.array([-G_phi, G_theta]))
    np.testing.assert_allclose(helmholtz[:, :, 0, :], -gradient)
    np.testing.assert_allclose(helmholtz[:, :, 1, :], rotated)
    np.testing.assert_allclose(laplacian_matrix, laplacian.to_matrix())

    evaluator = SphericalTransform(cs_basis, grid)
    np.testing.assert_allclose(evaluator.theta_derivative_array, G_theta)
    np.testing.assert_allclose(evaluator.phi_derivative_array, G_phi)


@pytest.mark.parametrize("basis_kind", ["CS", "SH"])
def test_helmholtz_divergence_and_radial_curl_are_laplacian_maps(basis_kind):
    """Helmholtz div/curl maps expose shared potential identities."""
    basis = GlobalCSBasis(8) if basis_kind == "CS" else SHBasis(3, 2)
    laplacian = to_numpy(basis.surface_laplacian_operator().to_matrix())
    curl_free_potential = to_numpy(basis.helmholtz_curl_free_potential_operator().to_array())
    divergence_free_potential = to_numpy(
        basis.helmholtz_divergence_free_potential_operator().to_array()
    )
    divergence = to_numpy(basis.helmholtz_surface_divergence_operator().to_array())
    radial_curl = to_numpy(basis.helmholtz_radial_curl_operator().to_array())
    identity = np.eye(basis.index_length)
    zeros = np.zeros_like(laplacian)

    assert laplacian.shape == (basis.index_length, basis.index_length)
    assert curl_free_potential.shape == (basis.index_length, 2, basis.index_length)
    assert divergence_free_potential.shape == (basis.index_length, 2, basis.index_length)
    assert divergence.shape == (basis.index_length, 2, basis.index_length)
    assert radial_curl.shape == (basis.index_length, 2, basis.index_length)
    np.testing.assert_allclose(curl_free_potential, np.stack([identity, zeros], axis=1))
    np.testing.assert_allclose(divergence_free_potential, np.stack([zeros, identity], axis=1))
    np.testing.assert_allclose(divergence, np.stack([-laplacian, zeros], axis=1))
    np.testing.assert_allclose(radial_curl, np.stack([zeros, laplacian], axis=1))

    rng = np.random.default_rng(20260521)
    coeffs = rng.standard_normal((2, basis.index_length))
    expected_curl_free = coeffs[0]
    expected_divergence_free = coeffs[1]
    expected_divergence = np.tensordot(divergence, coeffs, axes=([1, 2], [0, 1]))
    expected_radial_curl = np.tensordot(radial_curl, coeffs, axes=([1, 2], [0, 1]))

    actual_curl_free = basis.helmholtz_curl_free_potential_operator().matvec(coeffs.reshape(-1))
    actual_divergence_free = basis.helmholtz_divergence_free_potential_operator().matvec(
        coeffs.reshape(-1)
    )
    actual_divergence = basis.helmholtz_surface_divergence_operator().matvec(coeffs.reshape(-1))
    actual_radial_curl = basis.helmholtz_radial_curl_operator().matvec(coeffs.reshape(-1))
    np.testing.assert_allclose(to_numpy(actual_curl_free), expected_curl_free)
    np.testing.assert_allclose(to_numpy(actual_divergence_free), expected_divergence_free)
    np.testing.assert_allclose(to_numpy(actual_divergence), expected_divergence)
    np.testing.assert_allclose(to_numpy(actual_radial_curl), expected_radial_curl)


def test_solid_harmonics_match_reference_radius_shift_formulas():
    """Solid harmonics use geomagnetic reference-radius scaling."""
    sh_basis = SHBasis(3, 2)
    solid_harmonics = SolidHarmonicOperators(sh_basis)

    np.testing.assert_allclose(
        solid_harmonics.regular_reference_shift_factors(2.0, 3.0), (2.0 / 3.0) ** (1 - sh_basis.n)
    )
    np.testing.assert_allclose(
        solid_harmonics.irregular_reference_shift_factors(2.0, 3.0),
        (2.0 / 3.0) ** (sh_basis.n + 2),
    )
    np.testing.assert_allclose(
        solid_harmonics.poloidal_to_regular_potential_factors, -(sh_basis.n + 1)
    )
    np.testing.assert_allclose(solid_harmonics.poloidal_to_irregular_potential_factors, sh_basis.n)
    np.testing.assert_allclose(
        solid_harmonics.poloidal_to_normalized_potential_jump_factors, 2 * sh_basis.n + 1
    )
    np.testing.assert_allclose(
        solid_harmonics.poloidal_to_potential_jump_factors(7.0),
        7.0 * (2 * sh_basis.n + 1),
    )
    np.testing.assert_allclose(
        -sh_basis.n * solid_harmonics.poloidal_to_regular_potential_factors,
        sh_basis.n * (sh_basis.n + 1),
    )
    np.testing.assert_allclose(
        (sh_basis.n + 1) * solid_harmonics.poloidal_to_irregular_potential_factors,
        sh_basis.n * (sh_basis.n + 1),
    )


# Global cubed-sphere basis


def test_csbasis_evaluates_with_finite_difference_derivatives():
    """GlobalCSBasis exposes native finite-difference derivative matrices."""
    cs_basis = GlobalCSBasis(8)
    grid = type("GridLike", (), {"theta": cs_basis.mesh.theta, "phi": cs_basis.mesh.phi})()

    constant = np.ones(cs_basis.index_length)
    cos_theta = np.cos(np.deg2rad(cs_basis.mesh.theta))
    expected_dtheta = -np.sin(np.deg2rad(cs_basis.mesh.theta))

    G = cs_basis.scalar_evaluation_array(grid)
    G_theta = cs_basis.scalar_evaluation_array(grid, derivative="theta")

    np.testing.assert_allclose(G @ constant, constant)
    np.testing.assert_allclose(G_theta @ constant, 0.0, atol=1e-12)
    np.testing.assert_allclose(G_theta @ cos_theta, expected_dtheta, atol=1e-2)


def test_csbasis_native_grid_is_cell_centered_with_cell_areas():
    """Native CS coefficients live at cell centers with cell areas."""
    cs_basis = GlobalCSBasis(16)
    block, i, j = cs_basis.mesh.grid_line_indices()
    expected_xi = cs_basis.mesh.face_coordinate(i[:, :-1, :-1] + 0.5).reshape(-1)
    expected_eta = cs_basis.mesh.face_coordinate(j[:, :-1, :-1] + 0.5).reshape(-1)
    step = np.diff(cs_basis.mesh.face_coordinate(np.array([0, 1])))[0]
    midpoint_area = step**2 * np.sqrt(
        np.linalg.det(cs_basis.mesh.projection.metric_tensor(expected_xi, expected_eta))
    )

    np.testing.assert_allclose(cs_basis.mesh.xi, expected_xi)
    np.testing.assert_allclose(cs_basis.mesh.eta, expected_eta)
    np.testing.assert_array_equal(cs_basis.mesh.face, block[:, :-1, :-1].reshape(-1))
    assert np.all(cs_basis.mesh.cell_areas.reshape(-1) > 0.0)
    np.testing.assert_allclose(np.sum(cs_basis.mesh.cell_areas.reshape(-1)), 4 * np.pi)
    assert np.all(np.isfinite(cs_basis.mesh.theta))
    assert np.all(np.isfinite(cs_basis.mesh.phi))
    assert np.all(np.abs(np.sin(np.deg2rad(cs_basis.mesh.theta))) > 1e-12)
    assert (
        np.max(
            np.abs(cs_basis.mesh.cell_areas.reshape(-1) - midpoint_area)
            / cs_basis.mesh.cell_areas.reshape(-1)
        )
        < 1e-3
    )


def test_csbasis_local_metric_factors_match_gnomonic_mapping():
    """CS local metric factors are consistent with the gnomonic map."""
    cs_basis = GlobalCSBasis(16)
    xi, eta = cs_basis.mesh.xi, cs_basis.mesh.eta
    delta = cs_basis.mesh.projection.metric_delta(xi, eta)
    expected_sqrt_detg = 1.0 / (np.cos(xi) ** 2 * np.cos(eta) ** 2 * delta**1.5)
    g_covariant = cs_basis.mesh.projection.metric_tensor(xi, eta)
    g_contravariant = cs_basis.mesh.projection.metric_tensor(xi, eta, covariant=False)
    identity = np.einsum("nij,njk->nik", g_covariant, g_contravariant)
    expected_identity = np.broadcast_to(np.eye(3), identity.shape)

    np.testing.assert_allclose(cs_basis.mesh.sqrt_detg, expected_sqrt_detg)
    np.testing.assert_allclose(identity, expected_identity, atol=1e-12)


def test_csbasis_vector_coordinate_transforms_round_trip():
    """CS vector transform arrays are mutually consistent."""
    cs_basis = GlobalCSBasis(16)
    xi, eta, block = cs_basis.mesh.xi, cs_basis.mesh.eta, cs_basis.mesh.face
    identity = np.broadcast_to(np.eye(3), (cs_basis.index_length, 3, 3))

    pc = cs_basis.mesh.projection.cartesian_to_cube_vector_array(xi, eta, face=block)
    pc_inv = cs_basis.mesh.projection.cube_to_cartesian_vector_array(xi, eta, face=block)
    enu_to_cube = cs_basis.mesh.projection.enu_to_cube_vector_array(xi, eta, face=block)
    cube_to_enu = cs_basis.mesh.projection.cube_to_enu_vector_array(xi, eta, face=block)

    for removed_name in (
        "spherical_to_cube_vector_matrix",
        "cube_to_spherical_vector_matrix",
        "spherical_normalization_matrix",
    ):
        assert not hasattr(cs_basis.mesh.projection, removed_name)
    assert not hasattr(cs_basis, "interpolate_vector_components")

    np.testing.assert_allclose(np.einsum("nij,njk->nik", pc, pc_inv), identity, atol=1e-12)
    np.testing.assert_allclose(
        np.einsum("nij,njk->nik", enu_to_cube, cube_to_enu), identity, atol=1e-12
    )


def test_csbasis_non_native_scalar_evaluation_uses_interpolation():
    """CS scalar evaluation matches built-in interpolation."""
    cs_basis = GlobalCSBasis(8)
    _, theta, phi = cs_basis.mesh.projection.cube_to_spherical(
        cs_basis.mesh.face_coordinate(np.array([1.2, 2.3, 3.4, 4.5])),
        cs_basis.mesh.face_coordinate(np.array([1.1, 2.2, 3.1, 4.2])),
        np.zeros(4),
        degrees=True,
    )
    target = SphericalGrid(theta=theta, phi=phi)
    coeffs = np.sin(np.deg2rad(cs_basis.mesh.theta))

    G = cs_basis.scalar_evaluation_array(target)
    expected = cs_basis.interpolate_scalar(
        coeffs, cs_basis.mesh.theta, cs_basis.mesh.phi, target.theta, target.phi
    )

    np.testing.assert_allclose(G @ coeffs, expected)
    with pytest.raises(NotImplementedError, match="native cubed-sphere grid"):
        cs_basis.scalar_evaluation_array(target, derivative="theta")


def test_csbasis_non_native_helmholtz_uses_vector_interpolation():
    """CS non-native Helmholtz evaluation interpolates vectors."""
    cs_basis = GlobalCSBasis(8)
    native = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
    _, theta, phi = cs_basis.mesh.projection.cube_to_spherical(
        cs_basis.mesh.face_coordinate(np.array([1.2, 2.3, 3.4, 4.5])),
        cs_basis.mesh.face_coordinate(np.array([1.1, 2.2, 3.1, 4.2])),
        np.zeros(4),
        degrees=True,
    )
    target = SphericalGrid(theta=theta, phi=phi)

    rng = np.random.default_rng(20260520)
    coeffs = rng.standard_normal((2, cs_basis.index_length))
    native_helmholtz = cs_basis.helmholtz_synthesis_array(native)
    target_helmholtz = cs_basis.helmholtz_synthesis_array(target)
    native_vector = np.tensordot(native_helmholtz, coeffs, 2)
    actual = np.tensordot(target_helmholtz, coeffs, 2)

    expected_theta, expected_phi, _ = cs_basis.interpolate_vector(
        native_vector[0],
        native_vector[1],
        np.zeros_like(native_vector[0]),
        cs_basis.mesh.theta,
        cs_basis.mesh.phi,
        target.theta,
        target.phi,
    )
    expected = np.stack([expected_theta, expected_phi])

    assert target_helmholtz.shape == (2, target.size, 2, cs_basis.index_length)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_csbasis_multi_vector_interpolation_matches_per_field_calls():
    """CS vector interpolation supports multiple fields at once."""
    cs_basis = GlobalCSBasis(8)
    _, theta, phi = cs_basis.mesh.projection.cube_to_spherical(
        cs_basis.mesh.face_coordinate(np.array([1.2, 2.3, 3.4, 4.5])),
        cs_basis.mesh.face_coordinate(np.array([1.1, 2.2, 3.1, 4.2])),
        np.zeros(4),
        degrees=True,
    )
    fields_theta = np.stack(
        [np.sin(np.deg2rad(cs_basis.mesh.theta)), np.cos(np.deg2rad(cs_basis.mesh.phi))], axis=-1
    )
    fields_phi = np.stack(
        [np.cos(np.deg2rad(cs_basis.mesh.theta)), np.sin(np.deg2rad(cs_basis.mesh.phi))], axis=-1
    )
    fields_radial = np.zeros_like(fields_theta)

    multi = cs_basis.interpolate_vector(
        fields_theta,
        fields_phi,
        fields_radial,
        cs_basis.mesh.theta,
        cs_basis.mesh.phi,
        theta,
        phi,
    )
    per_field = [
        cs_basis.interpolate_vector(
            fields_theta[:, i],
            fields_phi[:, i],
            fields_radial[:, i],
            cs_basis.mesh.theta,
            cs_basis.mesh.phi,
            theta,
            phi,
        )
        for i in range(fields_theta.shape[-1])
    ]

    for component_index in range(3):
        expected = np.stack([field[component_index] for field in per_field], axis=-1)
        np.testing.assert_allclose(multi[component_index], expected)


# Analysis weights and regularization boundaries


def test_grid_basis_regularization_requires_a_natural_smoothness_norm():
    """Regularization declares basis-specific smoothness support."""
    cs_basis = GlobalCSBasis(8)
    evaluator = SphericalTransform(
        cs_basis, SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi), reg_lambda=1.0
    )

    with pytest.raises(NotImplementedError, match="surface smoothness weights"):
        _ = evaluator.scalar_regularization_operator
    with pytest.raises(NotImplementedError, match="surface smoothness weights"):
        _ = evaluator.helmholtz_regularization_operator


def test_area_weight_defaults_use_grid_areas_or_sin_theta():
    """Default area weights use CS areas or sin(theta) grid weights."""
    cs_basis = GlobalCSBasis(4)
    cs_grid = SphericalGrid(
        theta=cs_basis.mesh.theta,
        phi=cs_basis.mesh.phi,
        area_weights=cs_basis.mesh.cell_areas.reshape(-1),
    )
    regular_grid = SphericalGrid(
        theta=np.array([30.0, 90.0, 150.0]), phi=np.array([0.0, 90.0, 180.0])
    )

    np.testing.assert_allclose(
        grid_sqrt_area_weights(cs_grid), np.sqrt(cs_basis.mesh.cell_areas.reshape(-1))
    )
    np.testing.assert_allclose(
        grid_sqrt_area_weights(regular_grid), np.sqrt(np.sin(np.deg2rad(regular_grid.theta)))
    )


def test_area_weight_option_and_explicit_weights_override():
    """Global area weighting is used only without explicit weights."""
    cs_basis = GlobalCSBasis(4)
    grid = SphericalGrid(
        theta=cs_basis.mesh.theta,
        phi=cs_basis.mesh.phi,
        area_weights=cs_basis.mesh.cell_areas.reshape(-1),
    )
    explicit = np.linspace(1.0, 2.0, grid.size)

    unweighted = SphericalTransform(cs_basis, grid, area_weighted=False)
    weighted = SphericalTransform(cs_basis, grid, area_weighted=True)
    overridden = SphericalTransform(cs_basis, grid, sqrt_weights=explicit, area_weighted=True)

    assert unweighted.sqrt_weights is None
    np.testing.assert_allclose(
        weighted.sqrt_weights, np.sqrt(cs_basis.mesh.cell_areas.reshape(-1))
    )
    np.testing.assert_allclose(overridden.sqrt_weights, explicit)
    np.testing.assert_allclose(
        resolve_sqrt_weights(grid, area_weighted=True, vector=True),
        np.tile(np.sqrt(cs_basis.mesh.cell_areas.reshape(-1)), (2, 1)),
    )
    np.testing.assert_allclose(
        resolve_sqrt_weights(grid, sqrt_weights=explicit, area_weighted=True, vector=True),
        np.tile(explicit, (2, 1)),
    )


def test_weighted_tensor_pinv_matches_explicit_weighted_least_squares():
    """Weighted pseudoinverse solves weighted normal equations."""
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    sqrt_weights = np.array([1.0, 1.5, 2.0, 2.5])
    weight_matrix = np.diag(sqrt_weights**2)

    actual = weighted_tensor_pinv(A, sqrt_weights=sqrt_weights, n_leading_flattened=1)
    expected = np.linalg.solve(A.T @ weight_matrix @ A, A.T @ weight_matrix)

    np.testing.assert_allclose(actual, expected)


# Differential accuracy, scalar gauge, and caching


def test_csbasis_derivatives_match_first_spherical_harmonics():
    """CS derivatives match first-degree sphere functions."""
    cs_basis = GlobalCSBasis(8)
    grid = type("GridLike", (), {"theta": cs_basis.mesh.theta, "phi": cs_basis.mesh.phi})()
    theta = np.deg2rad(cs_basis.mesh.theta)
    phi = np.deg2rad(cs_basis.mesh.phi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    fields = [
        (x, np.cos(theta) * np.cos(phi), -np.sin(phi), -2 * x),
        (y, np.cos(theta) * np.sin(phi), np.cos(phi), -2 * y),
        (z, -np.sin(theta), np.zeros_like(theta), -2 * z),
    ]

    G_theta = cs_basis.scalar_evaluation_array(grid, derivative="theta")
    G_phi = cs_basis.scalar_evaluation_array(grid, derivative="phi")
    laplacian = cs_basis.surface_laplacian_operator()

    for values, expected_theta, expected_phi, expected_laplacian in fields:
        np.testing.assert_allclose(G_theta @ values, expected_theta, atol=1e-2)
        np.testing.assert_allclose(G_phi @ values, expected_phi, atol=1e-2)
        np.testing.assert_allclose(laplacian @ values, expected_laplacian, atol=1.2e-1)


def test_csbasis_derivative_convergence_rates_are_reasonable():
    """CS finite differences show expected RMS convergence rates."""

    def rms_errors(N):
        cs_basis = GlobalCSBasis(N)
        grid = type("GridLike", (), {"theta": cs_basis.mesh.theta, "phi": cs_basis.mesh.phi})()
        theta = np.deg2rad(cs_basis.mesh.theta)
        phi = np.deg2rad(cs_basis.mesh.phi)
        sin_theta = np.sin(theta)
        values_l1 = sin_theta * np.cos(phi)
        values_l2 = sin_theta**2 * np.cos(2 * phi)

        theta_error = cs_basis.scalar_evaluation_array(grid, derivative="theta") @ values_l1
        theta_error -= np.cos(theta) * np.cos(phi)
        phi_error = cs_basis.scalar_evaluation_array(grid, derivative="phi") @ values_l1
        phi_error -= -np.sin(phi)
        laplacian_l1_error = cs_basis.surface_laplacian_operator() @ values_l1
        laplacian_l1_error -= -2 * values_l1
        laplacian_l2_error = cs_basis.surface_laplacian_operator() @ values_l2
        laplacian_l2_error -= -6 * values_l2

        return np.array(
            [
                np.sqrt(np.mean(theta_error**2)),
                np.sqrt(np.mean(phi_error**2)),
                np.sqrt(np.mean(laplacian_l1_error**2)),
                np.sqrt(np.mean(laplacian_l2_error**2)),
            ]
        )

    resolutions = np.array([8, 12, 16])
    h = np.pi / (2 * resolutions)
    errors = np.array([rms_errors(int(N)) for N in resolutions])
    orders = [
        np.polyfit(np.log(h), np.log(errors[:, error_index]), 1)[0]
        for error_index in range(errors.shape[1])
    ]

    assert orders[0] > 1.9
    assert orders[1] > 1.9
    assert orders[2] > 1.4
    assert orders[3] > 1.8


def test_csbasis_mean_free_projection_is_area_weighted_and_operator_preserving():
    """CS scalar gauges use an area-weighted mean-free projection."""
    cs_basis = GlobalCSBasis(8)
    rng = np.random.default_rng(20260520)
    values = rng.standard_normal(cs_basis.index_length) + 3.0
    projected = cs_basis.project_scalar_mean_free(values)

    np.testing.assert_allclose(
        cs_basis.scalar_mean_weights,
        cs_basis.mesh.cell_areas.reshape(-1) / np.sum(cs_basis.mesh.cell_areas.reshape(-1)),
    )
    assert cs_basis.scalar_mean_weights is cs_basis.scalar_mean_weights
    assert not cs_basis.scalar_mean_weights.flags.writeable
    assert cs_basis.scalar_mean(projected) == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(
        cs_basis.surface_laplacian_operator() @ projected,
        cs_basis.surface_laplacian_operator() @ values,
        atol=1e-10,
    )

    grid = type("GridLike", (), {"theta": cs_basis.mesh.theta, "phi": cs_basis.mesh.phi})()
    helmholtz = np.stack([values, -2.0 * values + 0.5])
    projected_helmholtz = cs_basis.project_helmholtz_mean_free(helmholtz)

    np.testing.assert_allclose(cs_basis.scalar_mean(projected_helmholtz), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.tensordot(cs_basis.helmholtz_synthesis_array(grid), projected_helmholtz, 2),
        np.tensordot(cs_basis.helmholtz_synthesis_array(grid), helmholtz, 2),
        atol=1e-10,
    )


def test_shbasis_cache_controls_do_not_change_numerical_results():
    """SH grid caches are bounded, observable, and safely clearable."""
    basis = SHBasis(4, 3)
    grid = SphericalGrid(lat=[-60.0, 20.0, 70.0], lon=[0.0, 90.0, -120.0])
    expected = basis.surface_gradient_array(grid)
    _ = basis.surface_gradient_operator(grid)

    if isinstance(expected, np.ndarray):
        assert not expected.flags.writeable

    populated = basis.cache_info()
    assert populated["grids"] == 1
    assert populated["legendre_values"] > 0
    assert populated["arrays"] > 0
    assert populated["operators"] > 0
    assert populated["grids"] <= populated["max_size"]

    basis.clear_cache()
    assert basis.cache_info() == {
        "grids": 0,
        "legendre_values": 0,
        "arrays": 0,
        "operators": 0,
        "max_size": 8,
    }
    np.testing.assert_allclose(basis.surface_gradient_array(grid), expected)


def test_csbasis_cache_controls_do_not_change_numerical_results():
    """CS caches are bounded, observable, and safely clearable."""
    basis = GlobalCSBasis(4)
    grid = SphericalGrid(theta=basis.mesh.theta + 0.01, phi=basis.mesh.phi)
    expected = basis.scalar_evaluation_array(grid)

    populated = basis.cache_info()
    assert populated["surface_operators"] == 1
    assert populated["surface_operators"] <= populated["surface_max_size"]

    basis.clear_cache(shared_remaps=True)
    cleared = basis.cache_info()
    assert cleared["surface_operators"] == 0
    assert cleared["remap_operators"]["size"] == 0
    assert cleared["shared_remap_matrices"]["size"] == 0
    np.testing.assert_allclose(basis.scalar_evaluation_array(grid), expected)


def test_csbasis_reuses_one_unit_sphere_laplacian_at_all_radii():
    """Radius changes scale one sparse geometric Laplacian by 1/r²."""
    basis = GlobalCSBasis(4)
    values = np.linspace(-1.0, 1.0, basis.index_length)

    unit_values = basis.surface_laplacian_operator().matvec(values)
    unit_matrix = basis._unit_surface_laplacian_matrix
    radius_values = basis.surface_laplacian_operator(2.0).matvec(values)

    np.testing.assert_allclose(radius_values, unit_values / 4.0)
    assert basis._unit_surface_laplacian_matrix is unit_matrix
    assert basis.cache_info()["laplacian_built"]

    basis.clear_cache()
    assert not basis.cache_info()["laplacian_built"]


# Backend preservation


@pytest.mark.requires_jax
def test_csbasis_sparse_geometry_is_built_on_numpy(monkeypatch):
    """SciPy derivative geometry should not depend on active JAX arithmetic."""
    from kompe.cubed_sphere import global_differencing

    basis = GlobalCSBasis(4)
    original_face_coordinate = global_differencing.cs_coordinates.face_coordinate
    observed_backends = []

    def checked_face_coordinate(*args, **kwargs):
        observed_backends.append(get_backend())
        return original_face_coordinate(*args, **kwargs)

    previous_backend = jax_enabled()
    try:
        set_backend("jax")
        monkeypatch.setattr(
            global_differencing.cs_coordinates, "face_coordinate", checked_face_coordinate
        )
        basis.surface_laplacian_operator()
        assert get_backend() == "jax"
    finally:
        set_backend(previous_backend)

    assert observed_backends
    assert set(observed_backends) == {"numpy"}


@pytest.mark.requires_jax
def test_csbasis_mean_free_projection_preserves_jax_arrays():
    """CS gauge projection preserves backend arrays."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    try:
        set_backend("jax")
        cs_basis = GlobalCSBasis(4)
        values = jnp.asarray(np.arange(cs_basis.index_length, dtype=float))

        projected = cs_basis.project_scalar_mean_free(values)

        assert "jax" in type(projected).__module__
        numpy_values = to_numpy(values)
        scale = max(1.0, float(np.max(np.abs(numpy_values))))
        tolerance = 16 * np.finfo(numpy_values.dtype).eps * scale
        np.testing.assert_allclose(to_numpy(cs_basis.scalar_mean(projected)), 0.0, atol=tolerance)
    finally:
        set_backend(previous_backend)


@pytest.mark.requires_jax
def test_csbasis_surface_operators_preserve_jax_inputs():
    """CS surface operators accept backend arrays."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    try:
        set_backend("jax")
        cs_basis = GlobalCSBasis(4)
        grid = type(
            "GridLike",
            (),
            {"theta": jnp.asarray(cs_basis.mesh.theta), "phi": jnp.asarray(cs_basis.mesh.phi)},
        )()
        values = jnp.asarray(np.arange(cs_basis.index_length, dtype=float))

        G = cs_basis.scalar_evaluation_array(grid)
        laplacian_values = cs_basis.surface_laplacian_operator().matvec(values)

        assert "jax" in type(G).__module__
        assert "jax" in type(laplacian_values).__module__
        backend_dtype = to_numpy(values).dtype
        assert np.issubdtype(backend_dtype, np.floating)
        assert to_numpy(G).dtype == backend_dtype
        assert to_numpy(laplacian_values).dtype == backend_dtype
        np.testing.assert_allclose(
            to_numpy(laplacian_values),
            cs_basis.surface_laplacian_operator().to_matrix() @ to_numpy(values),
        )
    finally:
        set_backend(previous_backend)


@pytest.mark.requires_jax
def test_csbasis_surface_operator_cache_is_backend_neutral():
    """One structured CS remap serves NumPy and JAX inputs."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    try:
        set_backend("numpy")
        basis = GlobalCSBasis(4)
        grid = SphericalGrid(theta=basis.mesh.theta + 0.01, phi=basis.mesh.phi)
        operator = basis.scalar_evaluation_operator(grid)
        values = np.arange(basis.index_length, dtype=float)
        numpy_result = operator.matvec(values)

        set_backend("jax")
        cached_operator = basis.scalar_evaluation_operator(grid)
        jax_result = cached_operator.matvec(jnp.asarray(values))

        assert cached_operator is operator
        assert isinstance(numpy_result, np.ndarray)
        assert "jax" in type(jax_result).__module__
        np.testing.assert_allclose(to_numpy(jax_result), numpy_result)
    finally:
        set_backend(previous_backend)


@pytest.mark.requires_jax
def test_shbasis_surface_operators_preserve_jax_inputs():
    """SH surface operators accept backend arrays."""
    import jax.numpy as jnp

    previous_backend = jax_enabled()
    try:
        set_backend("jax")
        sh_basis = SHBasis(3, 2)
        grid = type(
            "GridLike",
            (),
            {
                "theta": jnp.asarray(np.array([30.0, 80.0])),
                "phi": jnp.asarray(np.array([0.0, 45.0])),
            },
        )()
        values = jnp.asarray(np.arange(sh_basis.index_length, dtype=float))

        G = sh_basis.scalar_evaluation_array(grid)
        grid_values = sh_basis.scalar_evaluation_operator(grid).matvec(values)
        shifted = (
            SolidHarmonicOperators(sh_basis)
            .regular_reference_shift_operator(2.0, 3.0)
            .matvec(values)
        )

        assert "jax" in type(G).__module__
        assert "jax" in type(sh_basis.surface_laplacian_operator().diagonal()).__module__
        assert "jax" in type(grid_values).__module__
        assert "jax" in type(shifted).__module__
    finally:
        set_backend(previous_backend)


# Mean-free and subset coefficient spaces


def test_shbasis_mean_free_option_matches_nmin_one_space():
    """Mean-free SH spaces match the min_degree=1 scalar space."""
    nmin_one = SHBasis(3, 2, min_degree=1)
    mean_free = SHBasis(3, 2, mean_free=True)
    full = SHBasis(3, 2, mean_free=False)
    cached_mean_free = full.with_mean_free(True)
    non_mean_free = mean_free.with_mean_free(False)

    assert isinstance(cached_mean_free, BasisSubset)
    assert isinstance(cached_mean_free, SurfaceDifferentialBasis)
    assert cached_mean_free.parent_basis is full
    assert cached_mean_free.root_basis is full
    assert mean_free.omits_constant_mode()
    assert mean_free.min_degree == nmin_one.min_degree == 1
    assert mean_free.index_length == nmin_one.index_length
    assert cached_mean_free.omits_constant_mode()
    assert cached_mean_free.with_mean_free(False) is full
    assert full.with_mean_free(True) is cached_mean_free
    assert not non_mean_free.omits_constant_mode()
    assert non_mean_free.min_degree == 0
    assert non_mean_free.index_length > mean_free.index_length


def test_shbasis_mean_free_view_slices_parent_operators():
    """Mean-free SH views slice the full parent coefficient space."""
    full = SHBasis(3, 2, mean_free=False)
    view = full.with_mean_free(True)
    direct_mean_free = SHBasis(3, 2, mean_free=True)
    grid = type("GridLike", (), {"theta": np.array([30.0, 80.0]), "phi": np.array([0.0, 45.0])})()

    assert view.index_length == direct_mean_free.index_length
    np.testing.assert_array_equal(view.index_arrays[0], direct_mean_free.index_arrays[0])
    np.testing.assert_array_equal(view.index_arrays[1], direct_mean_free.index_arrays[1])
    np.testing.assert_allclose(
        view.scalar_evaluation_array(grid), full.scalar_evaluation_array(grid)[:, 1:]
    )
    np.testing.assert_allclose(
        view.scalar_evaluation_array(grid), direct_mean_free.scalar_evaluation_array(grid)
    )
    np.testing.assert_allclose(
        view.surface_laplacian_operator().to_matrix(),
        direct_mean_free.surface_laplacian_operator().to_matrix(),
    )
    assert view.surface_laplacian_operator().is_diagonal
    view_solid_harmonics = SolidHarmonicOperators(view)
    direct_solid_harmonics = SolidHarmonicOperators(direct_mean_free)
    np.testing.assert_allclose(
        view_solid_harmonics.regular_reference_shift_factors(2.0, 3.0),
        direct_solid_harmonics.regular_reference_shift_factors(2.0, 3.0),
    )
    np.testing.assert_allclose(
        view_solid_harmonics.irregular_reference_shift_factors(2.0, 3.0),
        direct_solid_harmonics.irregular_reference_shift_factors(2.0, 3.0),
    )
    np.testing.assert_allclose(
        view_solid_harmonics.poloidal_to_normalized_potential_jump_factors,
        direct_solid_harmonics.poloidal_to_normalized_potential_jump_factors,
    )


def test_basis_view_slices_cs_surface_operators():
    """Generic basis subsets also slice CS coefficient-space operators."""
    cs_basis = GlobalCSBasis(8)
    indices = np.arange(0, cs_basis.index_length, 2)
    view = BasisSubset(cs_basis, indices, subset_name="even")
    grid = type("GridLike", (), {"theta": cs_basis.mesh.theta, "phi": cs_basis.mesh.phi})()

    assert isinstance(view, SurfaceDifferentialBasis)
    assert view.kind == "CS"
    assert view.index_length == indices.size
    np.testing.assert_allclose(view.index_arrays[0], cs_basis.mesh.theta[indices])
    np.testing.assert_allclose(view.index_arrays[1], cs_basis.mesh.phi[indices])
    np.testing.assert_allclose(
        view.scalar_evaluation_array(grid), cs_basis.scalar_evaluation_array(grid)[:, indices]
    )
    np.testing.assert_allclose(
        view.surface_gradient_array(grid),
        cs_basis.surface_gradient_array(grid)[:, :, indices],
    )
    expected = cs_basis.surface_laplacian_operator().to_matrix()[np.ix_(indices, indices)]
    np.testing.assert_allclose(view.surface_laplacian_operator().to_matrix(), expected)
    with pytest.raises(TypeError, match="SH surface basis"):
        SolidHarmonicOperators(view)


def test_shbasis_rejects_inconsistent_mean_free_options():
    """min_degree and mean_free must describe the same scalar space."""
    with pytest.raises(ValueError, match="inconsistent scalar-space options"):
        SHBasis(3, 2, min_degree=0, mean_free=True)


# Basis-owned operator caches


def test_spherical_transform_reuses_sh_evaluation_context(monkeypatch):
    """Basis-owned grid cache reuses expensive SH work."""
    sh_basis = SHBasis(4, 3)
    grid = SphericalGrid(theta=np.array([30.0, 65.0, 85.0]), phi=np.array([0.0, 45.0, 120.0]))
    transform = SphericalTransform(sh_basis, grid)
    calls = {"legendre": 0, "derivative": 0}
    original_legendre = sh_basis.legendre
    original_derivative = sh_basis.legendre_derivative

    def counted_legendre(theta):
        calls["legendre"] += 1
        return original_legendre(theta)

    def counted_derivative(theta, P):
        calls["derivative"] += 1
        return original_derivative(theta, P)

    monkeypatch.setattr(sh_basis, "legendre", counted_legendre)
    monkeypatch.setattr(sh_basis, "legendre_derivative", counted_derivative)

    _ = (
        transform.scalar_synthesis_array,
        transform.theta_derivative_array,
        transform.phi_derivative_array,
    )

    assert calls == {"legendre": 1, "derivative": 1}


def test_csbasis_reuses_native_operator_cache():
    """Native CS surface operators are cached on the basis."""
    cs_basis = GlobalCSBasis(8)
    grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
    same_grid = SphericalGrid(theta=cs_basis.mesh.theta.copy(), phi=cs_basis.mesh.phi.copy())

    assert cs_basis.scalar_evaluation_operator(grid) is cs_basis.scalar_evaluation_operator(
        same_grid
    )
    assert cs_basis.scalar_evaluation_operator(
        grid, derivative="theta"
    ) is cs_basis.scalar_evaluation_operator(same_grid, derivative="theta")
    assert cs_basis.surface_gradient_operator(grid) is cs_basis.surface_gradient_operator(
        same_grid
    )
    assert cs_basis.rhat_cross_gradient_operator(grid) is cs_basis.rhat_cross_gradient_operator(
        same_grid
    )
    assert cs_basis.helmholtz_synthesis_operator(grid) is cs_basis.helmholtz_synthesis_operator(
        same_grid
    )


# Abstract-interface enforcement


def test_incomplete_basis_subclass_is_rejected():
    """Subclasses must declare the required metadata fields."""

    class IncompleteBasis(ScalarBasis):
        kind = "incomplete"

    with pytest.raises(TypeError):
        IncompleteBasis()


def test_surface_operator_subclass_must_implement_evaluate_on_grid():
    """Surface-operator bases must define grid evaluation."""

    class IncompleteSurfaceOperators(SurfaceDifferentialBasis):
        kind = "incomplete"
        index_names = ("i",)
        index_length = 1
        index_arrays = ((0,),)

    with pytest.raises(TypeError):
        IncompleteSurfaceOperators()
