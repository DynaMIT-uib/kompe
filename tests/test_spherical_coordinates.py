"""Tests for spherical coordinates and local vector components."""

import numpy as np
import pytest

from kompe.math import backend_context
from kompe.spherical_coordinates import (
    cartesian_to_spherical,
    ecef_to_enu,
    enu_to_ecef,
    rotate_spherical_by_matrix,
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


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
@pytest.mark.parametrize("degrees", [True, False])
def test_spherical_conversions_broadcast_and_preserve_sample_axes(backend, degrees):
    radius = 2.0
    theta = np.array([[30.0], [90.0]])
    phi = np.array([0.0, 90.0, 270.0])
    if not degrees:
        theta, phi = np.deg2rad(theta), np.deg2rad(phi)
    with backend_context(backend):
        xyz = spherical_to_cartesian((radius, theta, phi), degrees=degrees)
        actual = cartesian_to_spherical(xyz, degrees=degrees)
        ring = spherical_to_cartesian((2.0, np.pi / 2, [0.0, np.pi / 2]), degrees=False)
        point = spherical_to_cartesian((2.0, 0.0, 0.0))
    expected = np.stack(np.broadcast_arrays(radius, theta, phi))
    assert xyz.shape == actual.shape == (3, 2, 3)
    assert point.shape == (3,)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(ring, [[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]], atol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_cartesian_conversion_preserves_small_polar_angles(backend):
    with backend_context(backend):
        result = cartesian_to_spherical(
            (np.array([1e-10, 1e-10]), 0.0, np.array([1.0, -1.0])), degrees=False
        )
    np.testing.assert_allclose(result[1, 0], 1e-10, rtol=1e-14)
    np.testing.assert_allclose(result[1, 1], np.pi - 1e-10, rtol=1e-14)
    with backend_context(backend):
        origin = cartesian_to_spherical((0.0, 0.0, 0.0))
    assert origin[0] == 0.0
    assert np.isnan(origin[1])


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

    actual = ecef_to_enu(enu_to_ecef(vectors, latitude, longitude), latitude, longitude)

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


def test_matrix_rotation_rotates_positions_and_tangent_vectors_together():
    quarter_turn_about_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    latitude = np.array([[0.0], [45.0]])
    longitude = np.array([0.0, 30.0, -120.0])
    east = np.ones((2, 3))
    north = np.arange(6.0).reshape(2, 3)

    actual = rotate_spherical_by_matrix(
        latitude,
        longitude,
        quarter_turn_about_z,
        east=east,
        north=north,
    )

    np.testing.assert_allclose(actual[0], np.broadcast_to(latitude, (2, 3)), atol=1e-14)
    expected_longitude = ((np.broadcast_to(longitude, (2, 3)) + 270.0) % 360.0) - 180.0
    np.testing.assert_allclose(actual[1], expected_longitude, atol=1e-14)
    np.testing.assert_allclose(actual[2], east, atol=1e-14)
    np.testing.assert_allclose(actual[3], north, atol=1e-14)


def test_enu_ecef_round_trip_preserves_broadcast_grid_shape():
    latitude = np.array([[0.0], [45.0]])
    longitude = np.array([0.0, 90.0, 140.0])
    vectors = np.arange(18.0).reshape(2, 3, 3)

    actual = ecef_to_enu(enu_to_ecef(vectors, latitude, longitude), latitude, longitude)

    np.testing.assert_allclose(actual, vectors, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)])
def test_geographic_rotations_accept_lists_and_broadcast_coordinates(backend):
    with backend_context(backend):
        ecef = enu_to_ecef([1.0, 2.0, 3.0], [10.0, 70.0], 20.0)
        enu = ecef_to_enu(ecef, [10.0, 70.0], 20.0)
        rotated = rotate_spherical_coordinates(10.0, [20.0, 30.0], 0.0, 0.0, 90.0, 0.0)
        transformed = rotate_spherical_by_matrix(
            [10.0, 70.0], 20.0, np.eye(3), east=[1.0, 2.0], north=3.0
        )
    np.testing.assert_allclose(enu, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], atol=1e-14)
    np.testing.assert_allclose(rotated, [[10.0, 10.0], [20.0, 30.0]], atol=1e-14)
    np.testing.assert_allclose(
        transformed, [[10.0, 70.0], [20.0, 20.0], [1.0, 2.0], [3.0, 3.0]], atol=1e-14
    )


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
        rotated = rotate_spherical_coordinates(latitude, longitude, 0.0, 0.0, 90.0, 0.0)
        matrix_rotated = rotate_spherical_by_matrix(
            latitude, longitude, jnp.eye(3), east=vectors[:, 0], north=vectors[:, 1]
        )

    for value in (cartesian, round_trip, ecef, enu, *rotated, *matrix_rotated):
        assert "jax" in type(value).__module__
    np.testing.assert_allclose(round_trip, spherical, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(enu, vectors, rtol=2e-12, atol=1e-14)
