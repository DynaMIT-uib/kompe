"""Tests for global cubed-sphere coordinate and vector transformations."""

import numpy as np

from kompe import GlobalCSProjection


def _interior_face_points():
    """Return deterministic points away from face boundaries and poles."""
    face = np.repeat(np.arange(6), 4)
    xi = np.tile(np.array([-0.55, -0.18, 0.21, 0.57]), 6)
    eta = np.tile(np.array([0.42, -0.31, 0.16, -0.47]), 6)
    radius = np.linspace(0.7, 1.6, face.size)
    return xi, eta, radius, face


def test_global_projection_coordinates_round_trip_on_every_face():
    """Cube, spherical, geographic, and Cartesian coordinates agree."""
    projection = GlobalCSProjection()
    xi, eta, radius, face = _interior_face_points()

    x, y, z = projection.cube_to_cartesian(xi, eta, radius=radius, face=face)
    actual_radius, theta, phi = projection.cube_to_spherical(xi, eta, radius=radius, face=face)
    longitude = np.rad2deg(phi)
    latitude = 90.0 - np.rad2deg(theta)
    actual_xi, actual_eta, actual_face = projection.geographic_to_cube(longitude, latitude, face)

    np.testing.assert_array_equal(projection.face_index(longitude, latitude), face)
    np.testing.assert_array_equal(actual_face, face)
    np.testing.assert_allclose(actual_xi, xi, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(actual_eta, eta, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(actual_radius, radius, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(x, radius * np.sin(theta) * np.cos(phi), rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(y, radius * np.sin(theta) * np.sin(phi), rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(z, radius * np.cos(theta), rtol=2e-14, atol=2e-15)


def test_global_projection_vector_components_round_trip_on_every_face():
    """Cube, Cartesian, and ENU component maps are mutually consistent."""
    projection = GlobalCSProjection()
    xi, eta, radius, face = _interior_face_points()
    angle = np.linspace(-1.2, 1.4, face.size)
    cartesian_vectors = np.stack(
        [np.cos(angle), np.sin(2.0 * angle), np.linspace(-0.8, 0.9, face.size)], axis=1
    )

    cartesian_to_cube = projection.cartesian_to_cube_vector_matrix(
        xi, eta, radius=radius, face=face
    )
    cube_to_cartesian = projection.cube_to_cartesian_vector_matrix(
        xi, eta, radius=radius, face=face
    )
    cube_to_enu = projection.cube_to_enu_vector_matrix(xi, eta, radius=radius, face=face)
    enu_to_cube = projection.enu_to_cube_vector_matrix(xi, eta, radius=radius, face=face)

    cube_vectors = np.einsum("nij,nj->ni", cartesian_to_cube, cartesian_vectors)
    recovered_cartesian = np.einsum("nij,nj->ni", cube_to_cartesian, cube_vectors)
    enu_vectors = np.einsum("nij,nj->ni", cube_to_enu, cube_vectors)
    recovered_cube = np.einsum("nij,nj->ni", enu_to_cube, enu_vectors)

    np.testing.assert_allclose(recovered_cartesian, cartesian_vectors, rtol=2e-14, atol=3e-15)
    np.testing.assert_allclose(recovered_cube, cube_vectors, rtol=2e-14, atol=3e-15)
    np.testing.assert_allclose(
        np.linalg.norm(enu_vectors, axis=1),
        np.linalg.norm(cartesian_vectors, axis=1),
        rtol=2e-14,
        atol=3e-15,
    )


def test_global_projection_coordinate_directions_are_tangential():
    """Unit xi and eta directions have no radial ENU component."""
    projection = GlobalCSProjection()
    xi, eta, radius, face = _interior_face_points()
    cube_to_enu = projection.cube_to_enu_vector_matrix(xi, eta, radius=radius, face=face)
    xi_direction = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (face.size, 3))
    eta_direction = np.broadcast_to(np.array([0.0, 1.0, 0.0]), (face.size, 3))

    xi_enu = np.einsum("nij,nj->ni", cube_to_enu, xi_direction)
    eta_enu = np.einsum("nij,nj->ni", cube_to_enu, eta_direction)

    np.testing.assert_allclose(xi_enu[:, 2], 0.0, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(eta_enu[:, 2], 0.0, rtol=0.0, atol=2e-15)
