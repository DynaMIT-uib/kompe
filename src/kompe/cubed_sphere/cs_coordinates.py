"""Coordinate transforms for the cubed-sphere grid."""

from __future__ import annotations

import numpy as np

from kompe.cubed_sphere.arrayutils import invert_3x3_matrices


def face_coordinate(index, N):
    """Return xi/eta coordinate values for grid-line indices."""
    if not isinstance(N, (int, np.integer)):
        raise TypeError("N must be an integer")
    if N < 1:
        raise ValueError("N must be at least 1")
    return -np.pi / 4 + index * np.pi / (2 * N)


def metric_delta(xi, eta):
    """Return the cubed-sphere metric delta parameter."""
    xi, eta = np.broadcast_arrays(xi, eta)
    return 1 + np.tan(xi) ** 2 + np.tan(eta) ** 2


def surface_metric_tensor(xi, eta, r=1):
    """Return covariant metric tensors on cubed-sphere faces."""
    xi, eta, r = map(np.ravel, np.broadcast_arrays(xi, eta, r))
    X = np.tan(xi)
    Y = np.tan(eta)
    sec2_xi = 1 + X**2
    sec2_eta = 1 + Y**2
    metric_delta = 1 + X**2 + Y**2
    common = r**2 * sec2_xi * sec2_eta / metric_delta**2

    metric = np.empty((xi.size, 2, 2))
    metric[:, 0, 0] = common * sec2_xi
    metric[:, 0, 1] = -common * X * Y
    metric[:, 1, 0] = metric[:, 0, 1]
    metric[:, 1, 1] = common * sec2_eta
    return metric


def metric_tensor(xi, eta, r=1, covariant=True):
    """Return cubed-sphere metric tensors."""
    xi, eta, r = map(np.ravel, np.broadcast_arrays(xi, eta, r))
    g = np.empty((xi.size, 3, 3))
    g[:, :2, :2] = surface_metric_tensor(xi, eta, r)
    g[:, 0, 2] = 0
    g[:, 1, 2] = 0
    g[:, 2, 0] = 0
    g[:, 2, 1] = 0
    g[:, 2, 2] = 1

    return g if covariant else invert_3x3_matrices(g)


def cube_to_cartesian(xi, eta, r=1, block=0):
    """Return Cartesian coordinates from CS coordinates."""
    xi, eta, r, block = np.broadcast_arrays(xi, eta, r, block)
    delta_value = metric_delta(xi, eta)
    x, y, z = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)

    mask = block == 0
    x[mask] = r[mask] / np.sqrt(delta_value[mask])
    y[mask] = r[mask] * np.tan(xi[mask]) / np.sqrt(delta_value[mask])
    z[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(delta_value[mask])

    mask = block == 1
    x[mask] = -r[mask] * np.tan(xi[mask]) / np.sqrt(delta_value[mask])
    y[mask] = r[mask] / np.sqrt(delta_value[mask])
    z[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(delta_value[mask])

    mask = block == 2
    x[mask] = -r[mask] / np.sqrt(delta_value[mask])
    y[mask] = -r[mask] * np.tan(xi[mask]) / np.sqrt(delta_value[mask])
    z[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(delta_value[mask])

    mask = block == 3
    x[mask] = r[mask] * np.tan(xi[mask]) / np.sqrt(delta_value[mask])
    y[mask] = -r[mask] / np.sqrt(delta_value[mask])
    z[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(delta_value[mask])

    mask = block == 4
    x[mask] = -r[mask] * np.tan(eta[mask]) / np.sqrt(delta_value[mask])
    y[mask] = r[mask] * np.tan(xi[mask]) / np.sqrt(delta_value[mask])
    z[mask] = r[mask] / np.sqrt(delta_value[mask])

    mask = block == 5
    x[mask] = r[mask] * np.tan(eta[mask]) / np.sqrt(delta_value[mask])
    y[mask] = r[mask] * np.tan(xi[mask]) / np.sqrt(delta_value[mask])
    z[mask] = -r[mask] / np.sqrt(delta_value[mask])

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


def face_index(lon, lat):
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


def geographic_to_cube(lon, lat, block=None):
    """Return CS coordinates for geocentric coordinates."""
    lon, lat = np.broadcast_arrays(lon, lat)
    shape = lon.shape
    size = lon.size

    if block is None:
        block = face_index(lon, lat)
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
    "cube_to_cartesian",
    "cube_to_spherical",
    "face_coordinate",
    "face_index",
    "geographic_to_cube",
    "metric_delta",
    "metric_tensor",
    "surface_metric_tensor",
]
