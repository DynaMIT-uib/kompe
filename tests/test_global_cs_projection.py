"""Tests for global cubed-sphere coordinate and vector transformations."""

import numpy as np
import pytest

from kompe import GlobalCSProjection
from kompe.math import backend_context, jit, to_jax, to_numpy
from kompe.spherical import enu_to_ecef


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


def test_global_projection_enu_basis_matches_ecef_convention():
    """The direct ENU bridge agrees with the shared geographic convention."""
    projection = GlobalCSProjection()
    xi, eta, radius, face = _interior_face_points()
    cube_to_cartesian = projection.cube_to_cartesian_vector_matrix(
        xi, eta, radius=radius, face=face
    )
    enu_to_cube = projection.enu_to_cube_vector_matrix(
        xi, eta, radius=radius, face=face
    )
    actual = np.einsum("nij,njk->nik", cube_to_cartesian, enu_to_cube)

    _, theta, longitude = projection.cube_to_spherical(
        xi, eta, radius=radius, face=face, degrees=True
    )
    enu_basis_vectors = np.tile(np.eye(3), (face.size, 1))
    expected = enu_to_ecef(
        enu_basis_vectors,
        np.repeat(90.0 - theta, 3),
        np.repeat(longitude, 3),
    ).reshape(face.size, 3, 3)
    expected = expected.transpose(0, 2, 1)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=3e-15)


def test_global_projection_enu_transforms_are_finite_at_poles():
    """Direct Cartesian ENU bases avoid longitude-coordinate singularities."""
    projection = GlobalCSProjection()
    radius = np.array([1.0, 2.0])
    face = np.array([4, 5])
    xi = np.zeros(2)
    eta = np.zeros(2)

    cube_to_enu = projection.cube_to_enu_vector_matrix(
        xi, eta, radius=radius, face=face
    )
    enu_to_cube = projection.enu_to_cube_vector_matrix(
        xi, eta, radius=radius, face=face
    )
    expected_cube_to_enu = np.zeros((2, 3, 3))
    expected_cube_to_enu[:, 0, 0] = radius
    expected_cube_to_enu[:, 1, 1] = radius
    expected_cube_to_enu[:, 2, 2] = 1.0

    assert np.isfinite(cube_to_enu).all()
    assert np.isfinite(enu_to_cube).all()
    np.testing.assert_allclose(cube_to_enu, expected_cube_to_enu, atol=3e-15)
    np.testing.assert_allclose(
        np.einsum("nij,njk->nik", cube_to_enu, enu_to_cube),
        np.broadcast_to(np.eye(3), (2, 3, 3)),
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


def test_cube_vector_matrix_is_the_coordinate_jacobian():
    """Vector columns are derivatives of the coordinate transformation."""
    projection = GlobalCSProjection()
    xi, eta, radius, face = _interior_face_points()
    jacobian = projection.cube_to_cartesian_vector_matrix(
        xi, eta, radius=radius, face=face
    )
    step = 1e-7

    derivatives = []
    for dxi, deta, dr in ((step, 0.0, 0.0), (0.0, step, 0.0), (0.0, 0.0, step)):
        plus = projection.cube_to_cartesian(
            xi + dxi,
            eta + deta,
            radius=radius + dr,
            face=face,
        )
        minus = projection.cube_to_cartesian(
            xi - dxi,
            eta - deta,
            radius=radius - dr,
            face=face,
        )
        derivatives.append((np.stack(plus, axis=1) - np.stack(minus, axis=1)) / (2 * step))

    finite_difference_jacobian = np.stack(derivatives, axis=2)
    np.testing.assert_allclose(jacobian, finite_difference_jacobian, rtol=2e-8, atol=2e-9)


@pytest.mark.requires_jax
def test_global_projection_algebra_stays_on_jax_backend():
    """Coordinate, metric, and vector algebra preserve device arrays."""
    projection = GlobalCSProjection()

    with backend_context("jax"):
        xi, eta, radius, face = (
            to_jax(values) for values in _interior_face_points()
        )
        longitude = to_jax(np.array([10.0, 100.0, -170.0, -80.0, 25.0, 25.0]))
        latitude = to_jax(np.array([10.0, 10.0, -10.0, -10.0, 70.0, -70.0]))
        cube_coordinates = projection.geographic_to_cube(longitude, latitude)
        cartesian = projection.cube_to_cartesian(xi, eta, radius=radius, face=face)
        spherical = projection.cube_to_spherical(xi, eta, radius=radius, face=face)
        arrays = (
            projection.metric_delta(xi, eta),
            projection.metric_tensor(xi, eta, radius=radius),
            projection.face_index(longitude, latitude),
            *cube_coordinates,
            *cartesian,
            *spherical,
            projection.cartesian_to_cube_vector_matrix(
                xi, eta, radius=radius, face=face
            ),
            projection.cube_to_cartesian_vector_matrix(
                xi, eta, radius=radius, face=face
            ),
            projection.enu_to_cube_vector_matrix(
                xi, eta, radius=radius, face=face
            ),
            projection.cube_to_enu_vector_matrix(
                xi, eta, radius=radius, face=face
            ),
            projection.face_to_face_vector_matrix(xi, eta, face, (face + 1) % 6),
        )
        compiled_cartesian = jit(
            lambda xi_values, eta_values, face_values: projection.cube_to_cartesian(
                xi_values,
                eta_values,
                face=face_values,
            )
        )(xi, eta, face)
        compiled_vector_matrix = jit(
            lambda xi_values, eta_values, radius_values, face_values: (
                projection.enu_to_cube_vector_matrix(
                    xi_values,
                    eta_values,
                    radius=radius_values,
                    face=face_values,
                )
            )
        )(xi, eta, radius, face)

    assert all("jax" in type(array).__module__ for array in arrays)
    assert all("jax" in type(array).__module__ for array in compiled_cartesian)
    assert "jax" in type(compiled_vector_matrix).__module__
    np.testing.assert_allclose(
        np.linalg.norm(np.column_stack(tuple(map(to_numpy, cartesian))), axis=1),
        to_numpy(radius),
        rtol=2e-14,
        atol=2e-15,
    )
