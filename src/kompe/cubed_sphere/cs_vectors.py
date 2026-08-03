"""Vector component transforms for cubed-sphere coordinates."""

from __future__ import annotations

import numpy as np

from kompe.cubed_sphere import cs_coordinates
from kompe.cubed_sphere.arrayutils import invert_3x3_matrices


def pc(xi, eta, r=1, block=0, inverse=False):
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

    return invert_3x3_matrices(pc) if inverse else pc


def ps(xi, eta, r=1, block=0, inverse=False):
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

    return invert_3x3_matrices(ps) if inverse else ps


def q_between_blocks(xi, eta, block_i, block_j):
    """Return component transforms between CS blocks."""
    xi_i, eta_i, block_i, block_j = map(np.ravel, np.broadcast_arrays(xi, eta, block_i, block_j))

    psi_inv = ps(xi_i, eta_i, r=1, block=block_i, inverse=True)
    _, theta, phi = cs_coordinates.cube_to_spherical(xi_i, eta_i, r=1, block=block_i, deg=True)
    xi_j, eta_j, _ = cs_coordinates.geo_to_cube(phi, 90 - theta, block=block_j)
    psj = ps(xi_j, eta_j, r=1, block=block_j)

    return np.einsum("nij, njk -> nik", psj, psi_inv)


def spherical_q(lat, r, inverse=False):
    """Return spherical component normalization matrices."""
    lat, r = map(np.ravel, np.broadcast_arrays(lat, r))

    q = np.zeros((lat.size, 3, 3), dtype=np.float64)
    q[:, 0, 0] = r * np.cos(np.deg2rad(lat))
    q[:, 1, 1] = r
    q[:, 2, 2] = 1

    return invert_3x3_matrices(q) if inverse else q


__all__ = ["pc", "ps", "q_between_blocks", "spherical_q"]
