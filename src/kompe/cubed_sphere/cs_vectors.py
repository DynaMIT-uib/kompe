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
    xi, eta, r, block = (value.reshape(-1) for value in xp.broadcast_arrays(xi, eta, r, block))
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
        dface_dxi / sqrt_delta[:, None] - face_position * (X * dX_dxi / delta**1.5)[:, None]
    )
    dunit_deta = (
        dface_deta / sqrt_delta[:, None] - face_position * (Y * dY_deta / delta**1.5)[:, None]
    )

    local_jacobian = xp.stack(
        (r[:, None] * dunit_dxi, r[:, None] * dunit_deta, unit_position),
        axis=2,
    )
    face_rotation = xp.asarray(cs_coordinates._FACE_TO_CARTESIAN)[block]
    return xp.einsum("nij,njk->nik", face_rotation, local_jacobian)


def _enu_to_cartesian_matrix(up):
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


def _face_to_face_matrix(xi, eta, block_i, block_j):
    """Return component transforms between CS blocks."""
    xp = get_array_module(xi, eta, block_i, block_j)
    xi_i, eta_i, block_i, block_j = (
        value.reshape(-1) for value in xp.broadcast_arrays(xi, eta, block_i, block_j)
    )

    source_to_cartesian = _cube_to_cartesian_matrix(xi_i, eta_i, block=block_i)
    _, theta, phi = cs_coordinates.cube_to_spherical(xi_i, eta_i, block_i, deg=True)
    xi_j, eta_j, _ = cs_coordinates.geographic_to_cube(phi, 90 - theta, block=block_j)
    cartesian_to_target = _cartesian_to_cube_matrix(xi_j, eta_j, block=block_j)

    return xp.einsum("nij,njk->nik", cartesian_to_target, source_to_cartesian)
