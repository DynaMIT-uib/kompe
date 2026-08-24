"""Vector component transforms for cubed-sphere coordinates."""

from __future__ import annotations

from kompe.cubed_sphere import cs_coordinates
from kompe.cubed_sphere.arrayutils import invert_3x3_matrices
from kompe.math.backend import get_array_module


def _cartesian_to_cube_matrix(xi, eta, r=1, block=0):
    """Return Cartesian-to-CS contravariant transform matrices."""
    return invert_3x3_matrices(_cube_to_cartesian_matrix(xi, eta, r=r, block=block))


def _cube_to_cartesian_matrix(xi, eta, r=1, block=0):
    """Return CS-to-Cartesian contravariant transform matrices.

    The columns are the coordinate basis vectors
    ``(d x/dxi, d x/deta, d x/dr)`` in ECEF components.
    """
    xp = get_array_module(xi, eta, r, block)
    xi, eta, r, block = (
        value.reshape(-1) for value in xp.broadcast_arrays(xi, eta, r, block)
    )
    block = block.astype(int)

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
        dface_dxi / sqrt_delta[:, None]
        - face_position * (X * dX_dxi / delta**1.5)[:, None]
    )
    dunit_deta = (
        dface_deta / sqrt_delta[:, None]
        - face_position * (Y * dY_deta / delta**1.5)[:, None]
    )

    local_jacobian = xp.stack(
        (r[:, None] * dunit_dxi, r[:, None] * dunit_deta, unit_position),
        axis=2,
    )
    face_rotation = xp.asarray(cs_coordinates._FACE_TO_CARTESIAN)[block]
    return xp.einsum("nij,njk->nik", face_rotation, local_jacobian)


def _geographic_coordinate_to_cartesian_matrix(xi, eta, r=1, block=0):
    """Return ``(longitude, latitude, radius)`` coordinate basis vectors."""
    xp = get_array_module(xi, eta, r, block)
    r, theta, phi = cs_coordinates.cube_to_spherical(
        xi,
        eta,
        block,
        r=r,
        deg=False,
    )
    r, theta, phi = (value.reshape(-1) for value in xp.broadcast_arrays(r, theta, phi))
    latitude = xp.pi / 2 - theta
    sin_latitude = xp.sin(latitude)
    cos_latitude = xp.cos(latitude)
    sin_longitude = xp.sin(phi)
    cos_longitude = xp.cos(phi)

    dposition_dlongitude = xp.stack(
        (
            -r * cos_latitude * sin_longitude,
            r * cos_latitude * cos_longitude,
            xp.zeros_like(r),
        ),
        axis=1,
    )
    dposition_dlatitude = xp.stack(
        (
            -r * sin_latitude * cos_longitude,
            -r * sin_latitude * sin_longitude,
            r * cos_latitude,
        ),
        axis=1,
    )
    dposition_dradius = xp.stack(
        (
            cos_latitude * cos_longitude,
            cos_latitude * sin_longitude,
            sin_latitude,
        ),
        axis=1,
    )
    return xp.stack(
        (dposition_dlongitude, dposition_dlatitude, dposition_dradius),
        axis=2,
    )


def _geographic_coordinate_to_cube_matrix(xi, eta, r=1, block=0):
    """Return geographic-coordinate-to-CS component matrices."""
    xp = get_array_module(xi, eta, r, block)
    cartesian_to_cube = _cartesian_to_cube_matrix(xi, eta, r=r, block=block)
    geographic_to_cartesian = _geographic_coordinate_to_cartesian_matrix(
        xi, eta, r=r, block=block
    )
    return xp.einsum("nij,njk->nik", cartesian_to_cube, geographic_to_cartesian)


def _cube_to_geographic_coordinate_matrix(xi, eta, r=1, block=0):
    """Return CS-to-geographic-coordinate component matrices."""
    return invert_3x3_matrices(
        _geographic_coordinate_to_cube_matrix(xi, eta, r=r, block=block)
    )


def _face_to_face_matrix(xi, eta, block_i, block_j):
    """Return component transforms between CS blocks."""
    xp = get_array_module(xi, eta, block_i, block_j)
    xi_i, eta_i, block_i, block_j = (
        value.reshape(-1)
        for value in xp.broadcast_arrays(xi, eta, block_i, block_j)
    )

    source_to_cartesian = _cube_to_cartesian_matrix(xi_i, eta_i, block=block_i)
    _, theta, phi = cs_coordinates.cube_to_spherical(
        xi_i, eta_i, block_i, deg=True
    )
    xi_j, eta_j, _ = cs_coordinates.geographic_to_cube(phi, 90 - theta, block=block_j)
    cartesian_to_target = _cartesian_to_cube_matrix(xi_j, eta_j, block=block_j)

    return xp.einsum("nij,njk->nik", cartesian_to_target, source_to_cartesian)


def _geographic_coordinate_to_enu_matrix(lat, r):
    """Return geographic-coordinate-to-ENU component matrices."""
    xp = get_array_module(lat, r)
    lat, r = (value.reshape(-1) for value in xp.broadcast_arrays(lat, r))
    scales = xp.stack(
        (r * xp.cos(xp.deg2rad(lat)), r, xp.ones_like(r)),
        axis=1,
    )
    return xp.eye(3, dtype=scales.dtype)[None, :, :] * scales[:, None, :]


def _enu_to_geographic_coordinate_matrix(lat, r):
    """Return ENU-to-geographic-coordinate component matrices."""
    return invert_3x3_matrices(_geographic_coordinate_to_enu_matrix(lat, r))
