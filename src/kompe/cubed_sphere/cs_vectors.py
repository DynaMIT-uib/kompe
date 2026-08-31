"""Vector component transforms for cubed-sphere coordinates."""

from __future__ import annotations

from kompe.cubed_sphere import cs_coordinates
from kompe.cubed_sphere.geometry_linalg import inverse_3x3
from kompe.math.backend import get_array_module


def _cartesian_to_cube_array(xi, eta, radius=1, face=0):
    """Return Cartesian-to-CS contravariant transforms at each point."""
    return inverse_3x3(_cube_to_cartesian_array(xi, eta, radius=radius, face=face))


def _cube_to_cartesian_array(xi, eta, radius=1, face=0):
    """Return CS-to-Cartesian contravariant transforms at each point.

    The columns are the coordinate basis vectors
    ``(d x/dxi, d x/deta, d x/dr)`` in ECEF components.
    """
    xp = get_array_module(xi, eta, radius, face)
    xi, eta, radius, face = (
        value.reshape(-1) for value in xp.broadcast_arrays(xi, eta, radius, face)
    )
    face = face.astype(int)

    X = xp.tan(xi)
    Y = xp.tan(eta)
    delta = 1 + X**2 + Y**2
    sqrt_delta = xp.sqrt(delta)
    face_position = xp.stack((xp.ones_like(X), X, Y), axis=1)
    unit_position = face_position / sqrt_delta[:, None]

    dX_dxi = 1 + X**2
    dY_deta = 1 + Y**2
    zeros = xp.zeros_like(X)
    dface_dxi = xp.stack((zeros, dX_dxi, zeros), axis=1)
    dface_deta = xp.stack((zeros, zeros, dY_deta), axis=1)
    dunit_dxi = (
        dface_dxi / sqrt_delta[:, None] - face_position * (X * dX_dxi / delta**1.5)[:, None]
    )
    dunit_deta = (
        dface_deta / sqrt_delta[:, None] - face_position * (Y * dY_deta / delta**1.5)[:, None]
    )

    local_jacobian = xp.stack(
        (radius[:, None] * dunit_dxi, radius[:, None] * dunit_deta, unit_position),
        axis=2,
    )
    face_rotation = xp.asarray(cs_coordinates._FACE_TO_CARTESIAN)[face]
    return xp.einsum("nij,njk->nik", face_rotation, local_jacobian)


def _enu_to_cartesian_array(up):
    """Return local east, north, and up basis vectors in ECEF components."""
    xp = get_array_module(up)
    up = xp.asarray(up)
    longitude = xp.arctan2(up[:, 1], up[:, 0])
    sin_longitude = xp.sin(longitude)
    cos_longitude = xp.cos(longitude)
    sin_colatitude = xp.hypot(up[:, 0], up[:, 1])
    cos_colatitude = up[:, 2]
    zeros = xp.zeros_like(longitude)

    east = xp.stack(
        (-sin_longitude, cos_longitude, zeros),
        axis=1,
    )
    north = xp.stack(
        (
            -cos_colatitude * cos_longitude,
            -cos_colatitude * sin_longitude,
            sin_colatitude,
        ),
        axis=1,
    )
    return xp.stack((east, north, up), axis=2)


def _face_to_face_array(xi, eta, source_face, target_face):
    """Return component transforms between cubed-sphere faces."""
    xp = get_array_module(xi, eta, source_face, target_face)
    source_xi, source_eta, source_face, target_face = (
        value.reshape(-1) for value in xp.broadcast_arrays(xi, eta, source_face, target_face)
    )

    source_to_cartesian = _cube_to_cartesian_array(source_xi, source_eta, face=source_face)
    _, theta, phi = cs_coordinates.cube_to_spherical(
        source_xi, source_eta, source_face, degrees=True
    )
    target_xi, target_eta, _ = cs_coordinates.geographic_to_cube(phi, 90 - theta, face=target_face)
    cartesian_to_target = _cartesian_to_cube_array(target_xi, target_eta, face=target_face)

    return xp.einsum("nij,njk->nik", cartesian_to_target, source_to_cartesian)
