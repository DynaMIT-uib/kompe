"""Tests for spherical coordinates and local vector components."""

import numpy as np
import pytest

from kompe.math import backend_context
from kompe.spherical import (
    cartesian_to_spherical,
    ecef_to_enu,
    enu_to_ecef,
    rotate_spherical_coordinates,
    spherical_to_cartesian,
)


@pytest.mark.parametrize("degrees", [True, False])
def test_spherical_cartesian_round_trip(degrees):
    spherical = np.array(
        [
            [1.0, 2.5, 6.0],
            [30.0, 75.0, 140.0],
            [10.0, 120.0, 300.0],
        ]
    )
    if not degrees:
        spherical[1:] = np.deg2rad(spherical[1:])

    actual = cartesian_to_spherical(
        spherical_to_cartesian(spherical, degrees=degrees), degrees=degrees
    )

    np.testing.assert_allclose(actual, spherical, rtol=1e-14, atol=1e-14)


def test_enu_ecef_round_trip_uses_latitude_longitude_order():
    vectors = np.array(
        [
            [1.0, 2.0, 3.0],
            [-4.0, 5.0, 0.5],
            [0.2, -0.7, 1.8],
        ]
    )
    latitude = np.array([0.0, 45.0, -60.0])
    longitude = np.array([0.0, 90.0, 140.0])

    actual = ecef_to_enu(
        enu_to_ecef(vectors, latitude, longitude), latitude, longitude
    )

    np.testing.assert_allclose(actual, vectors, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("degrees", [True, False])
def test_identity_rotated_frame_preserves_coordinates(degrees):
    latitude = np.array([-40.0, 10.0, 75.0])
    longitude = np.array([20.0, 120.0, 300.0])
    x_axis_latitude = 0.0
    x_axis_longitude = 0.0
    z_axis_latitude = 90.0
    z_axis_longitude = 0.0
    if not degrees:
        latitude = np.deg2rad(latitude)
        longitude = np.deg2rad(longitude)
        z_axis_latitude = np.pi / 2

    actual_latitude, actual_longitude = rotate_spherical_coordinates(
        latitude,
        longitude,
        x_axis_latitude,
        x_axis_longitude,
        z_axis_latitude,
        z_axis_longitude,
        degrees=degrees,
    )

    np.testing.assert_allclose(actual_latitude, latitude, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual_longitude, longitude, rtol=1e-14, atol=1e-14)


@pytest.mark.requires_jax
def test_spherical_coordinate_operations_preserve_jax_backend():
    import jax.numpy as jnp

    spherical = jnp.array([[1.0, 2.0], [30.0, 75.0], [10.0, 120.0]])
    vectors = jnp.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5]])
    latitude = jnp.array([10.0, 70.0])
    longitude = jnp.array([20.0, 120.0])

    with backend_context("jax"):
        cartesian = spherical_to_cartesian(spherical)
        round_trip = cartesian_to_spherical(cartesian)
        ecef = enu_to_ecef(vectors, latitude, longitude)
        enu = ecef_to_enu(ecef, latitude, longitude)
        rotated = rotate_spherical_coordinates(
            latitude, longitude, 0.0, 0.0, 90.0, 0.0
        )

    for value in (cartesian, round_trip, ecef, enu, *rotated):
        assert "jax" in type(value).__module__
    np.testing.assert_allclose(round_trip, spherical, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(enu, vectors, rtol=2e-12, atol=1e-14)

