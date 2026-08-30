"""Analytic checks for cubed-sphere surface differentiation."""

import numpy as np

from kompe import GlobalCSBasis, SphericalGrid


def _weighted_rms(values, weights):
    """Return component-averaged area-weighted RMS."""
    values = np.asarray(values)
    weights = np.asarray(weights) / np.sum(weights)
    components = values.reshape((-1, weights.size))
    return np.sqrt(np.mean(np.sum(weights * components**2, axis=1)))


def _analytic_surface_cases(theta_deg, phi_deg):
    """Return smooth fields and analytic surface derivatives."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    ax, ay, az = 0.35, -0.2, 0.15
    q = ax * sin_theta * np.cos(phi) + ay * sin_theta * np.sin(phi) + az * cos_theta
    exp_q = np.exp(q)
    q_theta = ax * cos_theta * np.cos(phi) + ay * cos_theta * np.sin(phi)
    q_theta -= az * sin_theta
    q_phi = -ax * np.sin(phi) + ay * np.cos(phi)

    return {
        "l1_x": (
            sin_theta * np.cos(phi),
            cos_theta * np.cos(phi),
            -np.sin(phi),
            -2 * sin_theta * np.cos(phi),
        ),
        "l1_y": (
            sin_theta * np.sin(phi),
            cos_theta * np.sin(phi),
            np.cos(phi),
            -2 * sin_theta * np.sin(phi),
        ),
        "l1_z": (cos_theta, -sin_theta, np.zeros_like(theta), -2 * cos_theta),
        "l2_cos2phi": (
            sin_theta**2 * np.cos(2 * phi),
            2 * sin_theta * cos_theta * np.cos(2 * phi),
            -2 * sin_theta * np.sin(2 * phi),
            -6 * sin_theta**2 * np.cos(2 * phi),
        ),
        "l2_sin2phi": (
            sin_theta**2 * np.sin(2 * phi),
            2 * sin_theta * cos_theta * np.sin(2 * phi),
            2 * sin_theta * np.cos(2 * phi),
            -6 * sin_theta**2 * np.sin(2 * phi),
        ),
        "l2_zonal": (
            3 * cos_theta**2 - 1,
            -6 * sin_theta * cos_theta,
            np.zeros_like(theta),
            -6 * (3 * cos_theta**2 - 1),
        ),
        "l2_cosphi": (
            sin_theta * cos_theta * np.cos(phi),
            (cos_theta**2 - sin_theta**2) * np.cos(phi),
            -cos_theta * np.sin(phi),
            -6 * sin_theta * cos_theta * np.cos(phi),
        ),
        "l2_sinphi": (
            sin_theta * cos_theta * np.sin(phi),
            (cos_theta**2 - sin_theta**2) * np.sin(phi),
            cos_theta * np.cos(phi),
            -6 * sin_theta * cos_theta * np.sin(phi),
        ),
        "exp_linear": (
            exp_q,
            exp_q * q_theta,
            exp_q * q_phi,
            exp_q * (ax**2 + ay**2 + az**2 - q**2 - 2 * q),
        ),
    }


def test_csbasis_differentiates_low_degree_spherical_harmonics():
    """CS derivative matrices match analytic l=1 and l=2 harmonics."""
    cs_basis = GlobalCSBasis(16)
    grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
    weights = cs_basis.mesh.cell_areas.reshape(-1)

    D_theta = cs_basis.scalar_evaluation_matrix(grid, derivative="theta")
    D_phi = cs_basis.scalar_evaluation_matrix(grid, derivative="phi")
    laplacian = cs_basis.surface_laplacian_operator()

    constant = np.ones(cs_basis.index_length)
    np.testing.assert_allclose(D_theta @ constant, 0.0, atol=1e-12)
    np.testing.assert_allclose(D_phi @ constant, 0.0, atol=1e-12)
    np.testing.assert_allclose(laplacian @ constant, 0.0, atol=1e-12)

    for values, expected_theta, expected_phi, expected_laplacian in _analytic_surface_cases(
        cs_basis.mesh.theta, cs_basis.mesh.phi
    ).values():
        gradient_error = _weighted_rms(
            np.stack([D_theta @ values - expected_theta, D_phi @ values - expected_phi]), weights
        )
        gradient_scale = _weighted_rms(np.stack([expected_theta, expected_phi]), weights)
        laplacian_error = _weighted_rms(laplacian @ values - expected_laplacian, weights)
        laplacian_scale = _weighted_rms(expected_laplacian, weights)

        assert gradient_error / gradient_scale < 7e-3
        assert laplacian_error / laplacian_scale < 2.5e-2


def test_csbasis_vector_surface_operators_match_analytic_composition():
    """Gradient, rotated gradient, and Helmholtz signs are correct."""
    cs_basis = GlobalCSBasis(16)
    grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
    weights = cs_basis.mesh.cell_areas.reshape(-1)
    cases = _analytic_surface_cases(cs_basis.mesh.theta, cs_basis.mesh.phi)

    f, f_theta, f_phi, _ = cases["l2_cos2phi"]
    g, g_theta, g_phi, _ = cases["l2_sinphi"]

    gradient = np.tensordot(cs_basis.surface_gradient_matrix(grid), f, axes=1)
    rotated_gradient = np.tensordot(cs_basis.rhat_cross_gradient_matrix(grid), f, axes=1)
    helmholtz = np.tensordot(cs_basis.helmholtz_synthesis_matrix(grid), np.stack([f, g]), axes=2)

    expected_gradient = np.stack([f_theta, f_phi])
    expected_rotated_gradient = np.stack([-f_phi, f_theta])
    expected_helmholtz = np.stack([-f_theta - g_phi, -f_phi + g_theta])

    for actual, expected in [
        (gradient, expected_gradient),
        (rotated_gradient, expected_rotated_gradient),
        (helmholtz, expected_helmholtz),
    ]:
        error = _weighted_rms(actual - expected, weights)
        scale = _weighted_rms(expected, weights)
        assert error / scale < 7e-3


def test_csbasis_differentiation_errors_converge_for_smooth_harmonics():
    """CS derivative errors converge for smooth l=1 and l=2 fields."""

    def aggregate_errors(N):
        cs_basis = GlobalCSBasis(N)
        grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
        weights = cs_basis.mesh.cell_areas.reshape(-1)

        D_theta = cs_basis.scalar_evaluation_matrix(grid, derivative="theta")
        D_phi = cs_basis.scalar_evaluation_matrix(grid, derivative="phi")
        laplacian = cs_basis.surface_laplacian_operator()

        gradient_errors = []
        laplacian_errors = []
        for values, expected_theta, expected_phi, expected_laplacian in _analytic_surface_cases(
            cs_basis.mesh.theta, cs_basis.mesh.phi
        ).values():
            gradient_errors.extend(
                [D_theta @ values - expected_theta, D_phi @ values - expected_phi]
            )
            laplacian_errors.append(laplacian @ values - expected_laplacian)

        return np.array(
            [
                _weighted_rms(np.stack(gradient_errors), weights),
                _weighted_rms(np.stack(laplacian_errors), weights),
            ]
        )

    resolutions = np.array([8, 12, 16, 24])
    h = np.pi / (2 * resolutions)
    errors = np.array([aggregate_errors(int(N)) for N in resolutions])
    orders = [
        np.polyfit(np.log(h), np.log(errors[:, error_index]), 1)[0]
        for error_index in range(errors.shape[1])
    ]

    assert orders[0] > 1.8
    assert orders[1] > 1.7


def test_csbasis_laplacian_integrates_to_zero_and_satisfies_green_identity():
    """CS areas and derivatives satisfy integral identities."""

    def residuals(N):
        cs_basis = GlobalCSBasis(N)
        grid = SphericalGrid(theta=cs_basis.mesh.theta, phi=cs_basis.mesh.phi)
        weights = cs_basis.mesh.cell_areas.reshape(-1)
        laplacian = cs_basis.surface_laplacian_operator()
        gradient = cs_basis.surface_gradient_matrix(grid)
        cases = _analytic_surface_cases(cs_basis.mesh.theta, cs_basis.mesh.phi)

        laplacian_integrals = []
        for values, *_ in cases.values():
            laplacian_values = laplacian @ values
            laplacian_integrals.append(
                abs(np.sum(weights * laplacian_values))
                / (np.sum(weights * np.abs(laplacian_values)) + 1e-30)
            )

        f = cases["l2_zonal"][0]
        g = cases["exp_linear"][0]
        grad_f = np.tensordot(gradient, f, axes=1)
        grad_g = np.tensordot(gradient, g, axes=1)
        f_laplacian_g = np.sum(weights * f * (laplacian @ g))
        grad_inner = np.sum(weights * (grad_f[0] * grad_g[0] + grad_f[1] * grad_g[1]))
        green_error = abs(f_laplacian_g + grad_inner)
        green_scale = abs(f_laplacian_g) + abs(grad_inner) + 1e-30

        return np.array(
            [np.sqrt(np.mean(np.asarray(laplacian_integrals) ** 2)), green_error / green_scale]
        )

    resolutions = np.array([8, 12, 16, 24])
    h = np.pi / (2 * resolutions)
    errors = np.array([residuals(int(N)) for N in resolutions])
    orders = [
        np.polyfit(np.log(h), np.log(errors[:, error_index]), 1)[0]
        for error_index in range(errors.shape[1])
    ]

    assert orders[0] > 1.7
    assert orders[1] > 1.7
