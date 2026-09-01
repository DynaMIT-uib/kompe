"""Coordinate transforms for the cubed-sphere grid."""

from __future__ import annotations

import numpy as np

from kompe.math.backend import get_array_module
from kompe.math.small_matrices import inverse_3x3

# Each matrix maps the canonical face frame ``(normal, xi, eta)`` to ECEF.
_FACE_TO_CARTESIAN = np.array(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    ],
    dtype=float,
)


def face_coordinate(index, cells_per_face):
    """Return xi/eta coordinate values for grid-line indices."""
    if not isinstance(cells_per_face, (int, np.integer)):
        raise TypeError("cells_per_face must be an integer")
    if cells_per_face < 1:
        raise ValueError("cells_per_face must be at least 1")
    xp = get_array_module(index)
    return -np.pi / 4 + xp.asarray(index) * np.pi / (2 * cells_per_face)


def metric_delta(xi, eta):
    """Return the cubed-sphere metric delta parameter."""
    xp = get_array_module(xi, eta)
    xi, eta = xp.broadcast_arrays(xp.asarray(xi), xp.asarray(eta))
    return 1 + xp.tan(xi) ** 2 + xp.tan(eta) ** 2


def surface_metric_tensor(xi, eta, radius=1):
    """Return covariant metric tensors on cubed-sphere faces."""
    xp = get_array_module(xi, eta, radius)
    xi, eta, radius = (value.reshape(-1) for value in xp.broadcast_arrays(xi, eta, radius))
    X = xp.tan(xi)
    Y = xp.tan(eta)
    sec2_xi = 1 + X**2
    sec2_eta = 1 + Y**2
    delta = 1 + X**2 + Y**2
    common = radius**2 * sec2_xi * sec2_eta / delta**2
    cross_term = -common * X * Y
    row_xi = xp.stack((common * sec2_xi, cross_term), axis=1)
    row_eta = xp.stack((cross_term, common * sec2_eta), axis=1)
    return xp.stack((row_xi, row_eta), axis=1)


def metric_tensor(xi, eta, radius=1, covariant=True):
    """Return cubed-sphere metric tensors."""
    xp = get_array_module(xi, eta, radius)
    surface = surface_metric_tensor(xi, eta, radius)
    zeros = xp.zeros(surface.shape[0], dtype=surface.dtype)
    row_xi = xp.stack((surface[:, 0, 0], surface[:, 0, 1], zeros), axis=1)
    row_eta = xp.stack((surface[:, 1, 0], surface[:, 1, 1], zeros), axis=1)
    row_radial = xp.stack((zeros, zeros, xp.ones_like(zeros)), axis=1)
    g = xp.stack((row_xi, row_eta, row_radial), axis=1)
    return g if covariant else inverse_3x3(g)


def cube_to_cartesian(xi, eta, radius=1, face=0):
    """Return Cartesian coordinates from CS coordinates."""
    xp = get_array_module(xi, eta, radius, face)
    xi, eta, radius, face = xp.broadcast_arrays(xi, eta, radius, face)
    shape = xi.shape
    xi, eta, radius = xi.reshape(-1), eta.reshape(-1), radius.reshape(-1)
    face = face.reshape(-1).astype(int)

    face_coordinates = xp.stack((xp.ones_like(xi), xp.tan(xi), xp.tan(eta)), axis=1)
    rotation = xp.asarray(_FACE_TO_CARTESIAN)[face]
    unit_cartesian = xp.einsum("nij,nj->ni", rotation, face_coordinates)
    cartesian = radius[:, None] * unit_cartesian / xp.sqrt(metric_delta(xi, eta))[:, None]
    return tuple(cartesian[:, component].reshape(shape) for component in range(3))


def cube_to_spherical(xi, eta, face, radius=1, degrees=False):
    """Return spherical coordinates from CS coordinates."""
    xp = get_array_module(xi, eta, radius, face)
    xi, eta, radius, face = xp.broadcast_arrays(xi, eta, radius, face)
    x, y, z = cube_to_cartesian(xi, eta, radius, face)
    phi = xp.arctan2(y, x)
    theta = xp.arccos(z / radius)

    if degrees:
        phi, theta = xp.rad2deg(phi), xp.rad2deg(theta)

    return radius, theta, phi


def face_index(lon, lat):
    """Return cube-face indices for geocentric coordinates."""
    xp = get_array_module(lon, lat)
    lon, lat = xp.broadcast_arrays(xp.asarray(lon), xp.asarray(lat))
    lon = xp.deg2rad(lon.reshape(-1))
    lat = xp.deg2rad(lat.reshape(-1))
    xyz = xp.stack(
        (xp.cos(lat) * xp.cos(lon), xp.cos(lat) * xp.sin(lon), xp.sin(lat)),
        axis=1,
    )
    face_normals = xp.asarray(_FACE_TO_CARTESIAN)[:, :, 0]
    return xp.argmax(xyz @ face_normals.T, axis=1)


def geographic_to_cube(lon, lat, face=None):
    """Return CS coordinates for geocentric coordinates."""
    xp = get_array_module(lon, lat, face)
    lon, lat = xp.broadcast_arrays(xp.asarray(lon), xp.asarray(lat))
    shape = lon.shape

    if face is None:
        face = face_index(lon, lat).reshape(shape)
    else:
        face = xp.broadcast_to(xp.asarray(face), shape)

    lon = xp.deg2rad(lon.reshape(-1))
    lat = xp.deg2rad(lat.reshape(-1))
    face = face.reshape(-1).astype(int)
    xyz = xp.stack(
        (xp.cos(lat) * xp.cos(lon), xp.cos(lat) * xp.sin(lon), xp.sin(lat)),
        axis=1,
    )
    rotation = xp.asarray(_FACE_TO_CARTESIAN)[face]
    face_coordinates = xp.einsum("nji,nj->ni", rotation, xyz)
    xi = xp.arctan(face_coordinates[:, 1] / face_coordinates[:, 0])
    eta = xp.arctan(face_coordinates[:, 2] / face_coordinates[:, 0])
    return xi.reshape(shape), eta.reshape(shape), face.reshape(shape)


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
