"""SECS signs and normalization checked directly against Biot-Savart."""

import numpy as np
import pytest
from scipy.integrate import quad_vec

from kompe.constants import MU0
from kompe.secs import (
    current_wedge_magnetic_field_matrices,
    magnetic_field_matrices,
    surface_current_matrices,
)


def test_spherical_df_field_matches_biot_savart_surface_integral():
    """A north-pole SECS carries K_phi=cot(theta/2)/(4 pi R)."""
    z, weights = np.polynomial.legendre.leggauss(100)
    longitude = np.arange(200) * (2 * np.pi / 200)
    z, longitude = np.meshgrid(z, longitude, indexing="ij")
    sin_theta = np.sqrt(1 - z**2)
    positions = np.stack(
        (sin_theta * np.cos(longitude), sin_theta * np.sin(longitude), z), axis=-1
    ).reshape(-1, 3)
    e_phi = np.stack((-np.sin(longitude), np.cos(longitude), np.zeros_like(z)), axis=-1).reshape(
        -1, 3
    )
    sheet_current = ((1 + z) / sin_theta / (4 * np.pi)).reshape(-1, 1) * e_phi
    area = np.broadcast_to(weights[:, None] * (2 * np.pi / 200), z.shape).reshape(-1)
    latitude, longitude, radius = (
        np.array([20.0, -35.0]),
        np.array([40.0, 110.0]),
        np.array([0.7, 1.4]),
    )
    lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
    up = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1)
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=1)
    north = np.cross(up, east)
    expected = []
    for i, position in enumerate(radius[:, None] * up):
        displacement = position - positions
        field = (
            MU0
            / (4 * np.pi)
            * np.sum(
                np.cross(sheet_current, displacement)
                * (area / np.linalg.norm(displacement, axis=1) ** 3)[:, None],
                axis=0,
            )
        )
        expected.append(np.stack((east[i], north[i], up[i])) @ field)
    actual = magnetic_field_matrices(latitude, longitude, radius, 90.0, 0.0, source_radius=1.0)
    np.testing.assert_allclose(np.asarray(actual)[:, :, 0].T, expected, rtol=1e-11, atol=1e-20)


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
def test_spherical_secs_fields_obey_ampere_jump(current_type):
    """In ENU, r-hat cross delta-B is (-delta-Bnorth, delta-Beast)."""
    lat, lon = [30.0, -10.0, 60.0], [20.0, 110.0, -40.0]
    current = np.asarray(
        surface_current_matrices(
            lat, lon, 75.0, 10.0, source_radius=1.0, current_type=current_type
        )
    )
    above = np.asarray(
        magnetic_field_matrices(
            lat, lon, 1.0 + 1e-9, 75.0, 10.0, source_radius=1.0, current_type=current_type
        )
    )
    below = np.asarray(
        magnetic_field_matrices(
            lat, lon, 1.0 - 1e-9, 75.0, 10.0, source_radius=1.0, current_type=current_type
        )
    )
    jump = above - below
    np.testing.assert_allclose(np.stack((-jump[1], jump[0])), MU0 * current, rtol=1e-8, atol=1e-18)
    np.testing.assert_allclose(jump[2], 0.0, atol=2e-15)


@pytest.mark.parametrize("direction_sign", [-1.0, 1.0])
def test_current_wedge_matches_biot_savart_line_integrals(direction_sign):
    """Positive current enters the sheet along the inclined ray and exits radially."""
    lat, lon, radius = (
        np.array([0.0, 30.0, -20.0]),
        np.array([0.0, 40.0, 120.0]),
        np.array([0.8, 1.3, 2.0]),
    )
    latitude, longitude = np.deg2rad(lat), np.deg2rad(lon)
    up = np.stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=1,
    )
    east = np.stack((-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)), axis=1)
    north = np.cross(up, east)
    endpoint = np.array([1.0, 0.0, 0.0])
    inclined = np.array([1.0, 0.3, 0.5])
    inclined /= np.linalg.norm(inclined)
    expected = []
    for i, position in enumerate(radius[:, None] * up):

        def integrand(s, position=position):
            along_inclined = position - endpoint - s * inclined
            along_radial = position - endpoint - s * endpoint
            return (
                -np.cross(inclined, along_inclined) / np.linalg.norm(along_inclined) ** 3
                + np.cross(endpoint, along_radial) / np.linalg.norm(along_radial) ** 3
            )

        field = MU0 / (4 * np.pi) * quad_vec(integrand, 0.0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
        expected.append(np.stack((east[i], north[i], up[i])) @ field)
    with np.errstate(divide="ignore", invalid="ignore"):
        actual = current_wedge_magnetic_field_matrices(
            lat,
            lon,
            radius,
            0.0,
            0.0,
            1.0,
            0.3 * direction_sign,
            0.5 * direction_sign,
            direction_sign,
        )
    np.testing.assert_allclose(np.asarray(actual)[:, :, 0].T, expected, rtol=1e-12, atol=1e-21)


@pytest.mark.parametrize("current_type", ["curl_free", "divergence_free"])
@pytest.mark.parametrize("radius", [1.0, 6.5e6])
def test_scalar_current_potential_gradient_recovers_current(current_type, radius):
    """Independent coordinate differences fix units, orientation, and both signs."""
    from kompe import SECSBasis, SphericalGrid

    basis = SECSBasis(
        SphericalGrid(lat=[65, -40, 12], lon=[20, 130, -65]),
        current_type=current_type,
        radius=radius,
    )
    theta = np.deg2rad([22, 50, 95, 150])
    phi = np.deg2rad([-35, 5, 80, -125])
    coefficients = np.array([0.7, -1.1, 0.4])
    step = 1e-5

    def potential(theta, phi):
        grid = SphericalGrid(theta=np.rad2deg(theta), phi=np.rad2deg(phi))
        return basis.scalar_evaluation_operator(grid)(coefficients)

    theta_gradient = (potential(theta + step, phi) - potential(theta - step, phi)) / (2 * step)
    phi_gradient = (potential(theta, phi + step) - potential(theta, phi - step)) / (
        2 * step * np.sin(theta)
    )
    gradient = np.stack([theta_gradient, phi_gradient])
    grid = SphericalGrid(theta=np.rad2deg(theta), phi=np.rad2deg(phi))
    for component, values in zip(("theta", "phi"), gradient, strict=True):
        np.testing.assert_allclose(
            basis.scalar_evaluation_operator(grid, gradient_component=component)(coefficients),
            values,
            rtol=2e-8,
            atol=2e-10,
        )
    expected = (
        -gradient / radius
        if current_type == "curl_free"
        else np.stack([-phi_gradient, theta_gradient]) / radius
    )
    np.testing.assert_allclose(
        basis.surface_current_operator(grid)(coefficients),
        expected,
        rtol=2e-8,
        atol=2e-10 / radius,
    )
