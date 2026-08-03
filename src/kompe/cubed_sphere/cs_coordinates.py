"""Coordinate transforms for the cubed-sphere grid."""

from __future__ import annotations

import numpy as np

from kompe.cubed_sphere.arrayutils import invert_3x3_matrices


def coordinate(index, N):
    """Return xi/eta coordinate values for grid-line indices."""
    if not isinstance(N, (int, np.integer)):
        raise TypeError("N must be an integer")
    if N < 1:
        raise ValueError("N must be at least 1")
    return -np.pi / 4 + index * np.pi / (2 * N)


def delta(xi, eta):
    """Return the cubed-sphere metric delta parameter."""
    xi, eta = np.broadcast_arrays(xi, eta)
    return 1 + np.tan(xi) ** 2 + np.tan(eta) ** 2


def metric_tensor(xi, eta, r=1, covariant=True):
    """Return cubed-sphere metric tensors."""
    xi, eta, r = map(np.ravel, np.broadcast_arrays(xi, eta, r))
    metric_delta = delta(xi, eta)

    g = np.empty((xi.size, 3, 3))
    g[:, 0, 0] = r**2 / (np.cos(xi) ** 4 * np.cos(eta) ** 2 * metric_delta**2)
    g[:, 0, 1] = (
        -(r**2) * np.tan(xi) * np.tan(eta) / (np.cos(xi) ** 2 * np.cos(eta) ** 2 * metric_delta**2)
    )
    g[:, 0, 2] = 0
    g[:, 1, 0] = (
        -(r**2) * np.tan(xi) * np.tan(eta) / (np.cos(xi) ** 2 * np.cos(eta) ** 2 * metric_delta**2)
    )
    g[:, 1, 1] = r**2 / (np.cos(xi) ** 2 * np.cos(eta) ** 4 * metric_delta**2)
    g[:, 1, 2] = 0
    g[:, 2, 0] = 0
    g[:, 2, 1] = 0
    g[:, 2, 2] = 1

    return g if covariant else invert_3x3_matrices(g)


def cube_to_cartesian(xi, eta, r=1, block=0):
    """Return Cartesian coordinates from CS coordinates."""
    xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)
    metric_delta = delta(xi, eta)
    x, y, z = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)

    mask = block == 0
    x[mask] = r[mask] / np.sqrt(metric_delta[mask])
    y[mask] = r[mask] * np.tan(xi[mask]) / np.sqrt(metric_delta[mask])
    z[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(metric_delta[mask])

    mask = block == 1
    x[mask] = -r[mask] * np.tan(xi[mask]) / np.sqrt(metric_delta[mask])
    y[mask] = r[mask] / np.sqrt(metric_delta[mask])
    z[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(metric_delta[mask])

    mask = block == 2
    x[mask] = -r[mask] / np.sqrt(metric_delta[mask])
    y[mask] = -r[mask] * np.tan(xi[mask]) / np.sqrt(metric_delta[mask])
    z[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(metric_delta[mask])

    mask = block == 3
    x[mask] = r[mask] * np.tan(xi[mask]) / np.sqrt(metric_delta[mask])
    y[mask] = -r[mask] / np.sqrt(metric_delta[mask])
    z[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(metric_delta[mask])

    mask = block == 4
    x[mask] = -r[mask] * np.tan(eta[mask]) / np.sqrt(metric_delta[mask])
    y[mask] = r[mask] * np.tan(xi[mask]) / np.sqrt(metric_delta[mask])
    z[mask] = r[mask] / np.sqrt(metric_delta[mask])

    mask = block == 5
    x[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(metric_delta[mask])
    y[mask] = r[mask] * np.tan(xi[mask]) / np.sqrt(metric_delta[mask])
    z[mask] = -r[mask] / np.sqrt(metric_delta[mask])

    return x, y, z


def cube_to_spherical(xi, eta, block, r=1, deg=False):
    """Return spherical coordinates from CS coordinates."""
    xi, eta = np.float64(xi), np.float64(eta)
    xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)

    x, y, z = cube_to_cartesian(xi, eta, r, block)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)

    if deg:
        phi, theta = np.rad2deg(phi), np.rad2deg(theta)

    return r, theta, phi


def cube_face(lon, lat):
    """Return cube-face indices for geocentric coordinates."""
    lon, lat = np.broadcast_arrays(lon, lat)
    lat, lon = lat.reshape(-1), lon.reshape(-1)

    theta, phi = np.deg2rad(90 - lat), np.deg2rad(lon)
    xyz = np.vstack((np.cos(phi) * np.sin(theta), np.sin(theta) * np.sin(phi), np.cos(theta)))
    face_midpoints = np.array(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )

    distances = np.empty((6, xyz.shape[1]))
    for index in range(6):
        distances[index] = np.linalg.norm(xyz - face_midpoints[index].reshape((3, 1)), axis=0)

    safety_distance = 1e-10
    blocks = np.zeros(xyz.shape[1], dtype=int)
    for index in range(6):
        blocks[distances[index] < np.choose(blocks, distances) - safety_distance] = index

    return blocks


def geo_to_cube(lon, lat, block=None):
    """Return CS coordinates for geocentric coordinates."""
    lon, lat = np.broadcast_arrays(lon, lat)
    shape = lon.shape
    size = lon.size

    if block is None:
        block = cube_face(lon, lat)
    else:
        block = block * np.ones_like(lat)

    block, lon, lat = block.reshape(-1), lon.reshape(-1), lat.reshape(-1)
    x, y, xi, eta = np.empty(size), np.empty(size), np.empty(size), np.empty(size)

    theta, phi = np.deg2rad(90 - lat), np.deg2rad(lon)
    x[block == 0] = np.tan(phi[block == 0])
    x[block == 1] = -1 / np.tan(phi[block == 1])
    x[block == 2] = np.tan(phi[block == 2])
    x[block == 3] = -1 / np.tan(phi[block == 3])
    x[block == 4] = np.tan(theta[block == 4]) * np.sin(phi[block == 4])
    x[block == 5] = -np.tan(theta[block == 5]) * np.sin(phi[block == 5])

    y[block == 0] = 1 / (np.tan(theta[block == 0]) * np.cos(phi[block == 0]))
    y[block == 1] = 1 / (np.tan(theta[block == 1]) * np.sin(phi[block == 1]))
    y[block == 2] = -1 / (np.tan(theta[block == 2]) * np.cos(phi[block == 2]))
    y[block == 3] = -1 / (np.tan(theta[block == 3]) * np.sin(phi[block == 3]))
    y[block == 4] = -np.tan(theta[block == 4]) * np.cos(phi[block == 4])
    y[block == 5] = -np.tan(theta[block == 5]) * np.cos(phi[block == 5])

    xi, eta = np.arctan(x), np.arctan(y)
    return xi.reshape(shape), eta.reshape(shape), block.reshape(shape)


__all__ = [
    "coordinate",
    "cube_face",
    "cube_to_cartesian",
    "cube_to_spherical",
    "delta",
    "geo_to_cube",
    "metric_tensor",
]
