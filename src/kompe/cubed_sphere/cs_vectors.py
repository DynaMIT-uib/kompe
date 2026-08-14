"""Vector component transforms for cubed-sphere coordinates."""

from __future__ import annotations

import numpy as np

from kompe.cubed_sphere import cs_coordinates
from kompe.cubed_sphere.arrayutils import invert_3x3_matrices


def _cartesian_to_cube_matrix(xi, eta, r=1, block=0):
    """Return Cartesian-to-CS contravariant transform matrices."""
    xi, et, r, block = map(np.ravel, np.broadcast_arrays(xi, eta, r, block))
    delta = cs_coordinates.delta(xi, et)
    pc = np.empty((delta.size, 3, 3))

    rsec2xi = r / np.cos(xi) ** 2
    rsec2et = r / np.cos(et) ** 2

    mask = block == 0
    pc[mask, 0, 0] = -np.sqrt(delta[mask]) * np.tan(xi[mask]) / rsec2xi[mask]
    pc[mask, 0, 1] = np.sqrt(delta[mask]) / rsec2xi[mask]
    pc[mask, 0, 2] = 0
    pc[mask, 1, 0] = -np.sqrt(delta[mask]) * np.tan(et[mask]) / rsec2et[mask]
    pc[mask, 1, 1] = 0
    pc[mask, 1, 2] = np.sqrt(delta[mask]) / rsec2et[mask]
    pc[mask, 2, 0] = 1 / np.sqrt(delta[mask])
    pc[mask, 2, 1] = np.tan(xi[mask]) / np.sqrt(delta[mask])
    pc[mask, 2, 2] = np.tan(et[mask]) / np.sqrt(delta[mask])

    mask = block == 1
    pc[mask, 0, 0] = -np.sqrt(delta[mask]) / rsec2xi[mask]
    pc[mask, 0, 1] = -np.sqrt(delta[mask]) * np.tan(xi[mask]) / rsec2xi[mask]
    pc[mask, 0, 2] = 0
    pc[mask, 1, 0] = 0
    pc[mask, 1, 1] = -np.sqrt(delta[mask]) * np.tan(et[mask]) / rsec2et[mask]
    pc[mask, 1, 2] = np.sqrt(delta[mask]) / rsec2et[mask]
    pc[mask, 2, 0] = -np.tan(xi[mask]) / np.sqrt(delta[mask])
    pc[mask, 2, 1] = 1 / np.sqrt(delta[mask])
    pc[mask, 2, 2] = np.tan(et[mask]) / np.sqrt(delta[mask])

    mask = block == 2
    pc[mask, 0, 0] = np.sqrt(delta[mask]) * np.tan(xi[mask]) / rsec2xi[mask]
    pc[mask, 0, 1] = -np.sqrt(delta[mask]) / rsec2xi[mask]
    pc[mask, 0, 2] = 0
    pc[mask, 1, 0] = np.sqrt(delta[mask]) * np.tan(et[mask]) / rsec2et[mask]
    pc[mask, 1, 1] = 0
    pc[mask, 1, 2] = np.sqrt(delta[mask]) / rsec2et[mask]
    pc[mask, 2, 0] = -1 / np.sqrt(delta[mask])
    pc[mask, 2, 1] = -np.tan(xi[mask]) / np.sqrt(delta[mask])
    pc[mask, 2, 2] = np.tan(et[mask]) / np.sqrt(delta[mask])

    mask = block == 3
    pc[mask, 0, 0] = np.sqrt(delta[mask]) / rsec2xi[mask]
    pc[mask, 0, 1] = np.sqrt(delta[mask]) * np.tan(xi[mask]) / rsec2xi[mask]
    pc[mask, 0, 2] = 0
    pc[mask, 1, 0] = 0
    pc[mask, 1, 1] = np.sqrt(delta[mask]) * np.tan(et[mask]) / rsec2et[mask]
    pc[mask, 1, 2] = np.sqrt(delta[mask]) / rsec2et[mask]
    pc[mask, 2, 0] = np.tan(xi[mask]) / np.sqrt(delta[mask])
    pc[mask, 2, 1] = -1 / np.sqrt(delta[mask])
    pc[mask, 2, 2] = np.tan(et[mask]) / np.sqrt(delta[mask])

    mask = block == 4
    pc[mask, 0, 0] = 0
    pc[mask, 0, 1] = np.sqrt(delta[mask]) / rsec2xi[mask]
    pc[mask, 0, 2] = -np.sqrt(delta[mask]) * np.tan(xi[mask]) / rsec2xi[mask]
    pc[mask, 1, 0] = -np.sqrt(delta[mask]) / rsec2et[mask]
    pc[mask, 1, 1] = 0
    pc[mask, 1, 2] = -np.sqrt(delta[mask]) * np.tan(et[mask]) / rsec2et[mask]
    pc[mask, 2, 0] = -np.tan(et[mask]) / np.sqrt(delta[mask])
    pc[mask, 2, 1] = np.tan(xi[mask]) / np.sqrt(delta[mask])
    pc[mask, 2, 2] = 1 / np.sqrt(delta[mask])

    mask = block == 5
    pc[mask, 0, 0] = 0
    pc[mask, 0, 1] = np.sqrt(delta[mask]) / rsec2xi[mask]
    pc[mask, 0, 2] = np.sqrt(delta[mask]) * np.tan(xi[mask]) / rsec2xi[mask]
    pc[mask, 1, 0] = np.sqrt(delta[mask]) / rsec2et[mask]
    pc[mask, 1, 1] = 0
    pc[mask, 1, 2] = np.sqrt(delta[mask]) * np.tan(et[mask]) / rsec2et[mask]
    pc[mask, 2, 0] = np.tan(et[mask]) / np.sqrt(delta[mask])
    pc[mask, 2, 1] = np.tan(xi[mask]) / np.sqrt(delta[mask])
    pc[mask, 2, 2] = -1 / np.sqrt(delta[mask])

    return pc


def _cube_to_cartesian_matrix(xi, eta, r=1, block=0):
    """Return CS-to-Cartesian contravariant transform matrices."""
    return invert_3x3_matrices(_cartesian_to_cube_matrix(xi, eta, r=r, block=block))


def _spherical_coordinate_to_cube_matrix(xi, eta, r=1, block=0):
    """Return spherical-to-CS contravariant transform matrices."""
    xi, et, r, block = map(np.ravel, np.broadcast_arrays(xi, eta, r, block))
    delta = cs_coordinates.delta(xi, et)
    ps = np.empty((delta.size, 3, 3))

    mask = block == 0
    ps[mask, 0, 0] = 1
    ps[mask, 0, 1] = 0
    ps[mask, 0, 2] = 0
    ps[mask, 1, 0] = np.tan(xi[mask]) * np.sin(et[mask]) * np.cos(et[mask])
    ps[mask, 1, 1] = np.cos(xi[mask]) * np.sin(et[mask]) ** 2 + np.cos(et[mask]) ** 2 / np.cos(
        xi[mask]
    )
    ps[mask, 1, 2] = 0
    ps[mask, 2, 0] = 0
    ps[mask, 2, 1] = 0
    ps[mask, 2, 2] = 1

    mask = block == 1
    ps[mask, 0, 0] = 1
    ps[mask, 0, 1] = 0
    ps[mask, 0, 2] = 0
    ps[mask, 1, 0] = np.tan(xi[mask]) * np.sin(et[mask]) * np.cos(et[mask])
    ps[mask, 1, 1] = np.cos(xi[mask]) * np.sin(et[mask]) ** 2 + np.cos(et[mask]) ** 2 / np.cos(
        xi[mask]
    )
    ps[mask, 1, 2] = 0
    ps[mask, 2, 0] = 0
    ps[mask, 2, 1] = 0
    ps[mask, 2, 2] = 1

    mask = block == 2
    ps[mask, 0, 0] = 1
    ps[mask, 0, 1] = 0
    ps[mask, 0, 2] = 0
    ps[mask, 1, 0] = np.tan(xi[mask]) * np.sin(et[mask]) * np.cos(et[mask])
    ps[mask, 1, 1] = np.cos(xi[mask]) * np.sin(et[mask]) ** 2 + np.cos(et[mask]) ** 2 / np.cos(
        xi[mask]
    )
    ps[mask, 1, 2] = 0
    ps[mask, 2, 0] = 0
    ps[mask, 2, 1] = 0
    ps[mask, 2, 2] = 1

    mask = block == 3
    ps[mask, 0, 0] = 1
    ps[mask, 0, 1] = 0
    ps[mask, 0, 2] = 0
    ps[mask, 1, 0] = np.tan(xi[mask]) * np.sin(et[mask]) * np.cos(et[mask])
    ps[mask, 1, 1] = np.cos(xi[mask]) * np.sin(et[mask]) ** 2 + np.cos(et[mask]) ** 2 / np.cos(
        xi[mask]
    )
    ps[mask, 1, 2] = 0
    ps[mask, 2, 0] = 0
    ps[mask, 2, 1] = 0
    ps[mask, 2, 2] = 1

    mask = block == 4
    ps[mask, 0, 0] = -(np.cos(xi[mask]) ** 2) * np.tan(et[mask])
    ps[mask, 0, 1] = (
        -delta[mask] * np.tan(xi[mask]) * np.cos(xi[mask]) ** 2 / np.sqrt(delta[mask] - 1)
    )
    ps[mask, 0, 2] = 0
    ps[mask, 1, 0] = np.cos(et[mask]) ** 2 * np.tan(xi[mask])
    ps[mask, 1, 1] = (
        -delta[mask] * np.tan(et[mask]) * np.cos(et[mask]) ** 2 / np.sqrt(delta[mask] - 1)
    )
    ps[mask, 1, 2] = 0
    ps[mask, 2, 0] = 0
    ps[mask, 2, 1] = 0
    ps[mask, 2, 2] = 1

    mask = block == 5
    ps[mask, 0, 0] = np.cos(xi[mask]) ** 2 * np.tan(et[mask])
    ps[mask, 0, 1] = (
        delta[mask] * np.tan(xi[mask]) * np.cos(xi[mask]) ** 2 / np.sqrt(delta[mask] - 1)
    )
    ps[mask, 0, 2] = 0
    ps[mask, 1, 0] = -(np.cos(et[mask]) ** 2) * np.tan(xi[mask])
    ps[mask, 1, 1] = (
        delta[mask] * np.tan(et[mask]) * np.cos(et[mask]) ** 2 / np.sqrt(delta[mask] - 1)
    )
    ps[mask, 1, 2] = 0
    ps[mask, 2, 0] = 0
    ps[mask, 2, 1] = 0
    ps[mask, 2, 2] = 1

    return ps


def _cube_to_spherical_coordinate_matrix(xi, eta, r=1, block=0):
    """Return CS-to-spherical-coordinate component matrices."""
    return invert_3x3_matrices(_spherical_coordinate_to_cube_matrix(xi, eta, r=r, block=block))


def _face_to_face_matrix(xi, eta, block_i, block_j):
    """Return component transforms between CS blocks."""
    xi_i, eta_i, block_i, block_j = map(np.ravel, np.broadcast_arrays(xi, eta, block_i, block_j))

    source_to_spherical = _cube_to_spherical_coordinate_matrix(xi_i, eta_i, r=1, block=block_i)
    _, theta, phi = cs_coordinates.cube_to_spherical(xi_i, eta_i, r=1, block=block_i, deg=True)
    xi_j, eta_j, _ = cs_coordinates.geo_to_cube(phi, 90 - theta, block=block_j)
    spherical_to_target = _spherical_coordinate_to_cube_matrix(xi_j, eta_j, r=1, block=block_j)

    return np.einsum("nij, njk -> nik", spherical_to_target, source_to_spherical)


def _spherical_coordinate_to_enu_matrix(lat, r):
    """Return spherical component normalization matrices."""
    lat, r = map(np.ravel, np.broadcast_arrays(lat, r))

    q = np.zeros((lat.size, 3, 3), dtype=np.float64)
    q[:, 0, 0] = r * np.cos(np.deg2rad(lat))
    q[:, 1, 1] = r
    q[:, 2, 2] = 1

    return q


def _enu_to_spherical_coordinate_matrix(lat, r):
    """Return ENU-to-spherical-coordinate component matrices."""
    return invert_3x3_matrices(_spherical_coordinate_to_enu_matrix(lat, r))
