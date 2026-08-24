"""Spherical-coordinate conversions shared across Kompe."""

import numpy as np

from kompe.math.backend import get_array_module

DEGREES_TO_RADIANS = np.pi / 180
RADIANS_TO_DEGREES = 180 / np.pi


def spherical_to_cartesian(spherical_coordinates, degrees=True):
    """Convert ``(radius, colatitude, longitude)`` to Cartesian coordinates."""
    r, theta, phi = spherical_coordinates
    xp = get_array_module(r, theta, phi)
    r, theta, phi = (xp.asarray(value) for value in (r, theta, phi))

    if not degrees:
        conv = 1.0
    else:
        conv = DEGREES_TO_RADIANS

    return xp.vstack(
        (
            r * xp.sin(theta * conv) * xp.cos(phi * conv),
            r * xp.sin(theta * conv) * xp.sin(phi * conv),
            r * xp.cos(theta * conv),
        )
    )


def cartesian_to_spherical(cartesian_coordinates, degrees=True):
    """Convert Cartesian coordinates to ``(radius, colatitude, longitude)``."""
    x, y, z = cartesian_coordinates
    xp = get_array_module(x, y, z)
    x, y, z = (xp.asarray(value) for value in (x, y, z))

    if not degrees:
        conv = 1.0
    else:
        conv = RADIANS_TO_DEGREES

    r = xp.sqrt(x**2 + y**2 + z**2)
    theta = xp.arccos(z / r) * conv
    phi = xp.mod(xp.arctan2(y, x), 2 * np.pi) * conv

    return xp.vstack((r, theta, phi))


def rotate_spherical_coordinates(
    latitude,
    longitude,
    x_axis_latitude,
    x_axis_longitude,
    z_axis_latitude,
    z_axis_longitude,
    degrees=True,
):
    """Express geographic coordinates in a rotated spherical frame."""
    xp = get_array_module(latitude, longitude)
    latitude = xp.asarray(latitude).flatten()
    longitude = xp.asarray(longitude).flatten()

    if not degrees:
        conv = 1.0
    else:
        conv = DEGREES_TO_RADIANS

    xyz = xp.vstack(
        (
            xp.cos(latitude * conv) * xp.cos(longitude * conv),
            xp.cos(latitude * conv) * xp.sin(longitude * conv),
            xp.sin(latitude * conv),
        )
    )

    new_z = np.array(
        [
            np.cos(z_axis_latitude * conv) * np.cos(z_axis_longitude * conv),
            np.cos(z_axis_latitude * conv) * np.sin(z_axis_longitude * conv),
            np.sin(z_axis_latitude * conv),
        ]
    )
    new_x = np.array(
        [
            np.cos(x_axis_latitude * conv) * np.cos(x_axis_longitude * conv),
            np.cos(x_axis_latitude * conv) * np.sin(x_axis_longitude * conv),
            np.sin(x_axis_latitude * conv),
        ]
    )
    new_y = np.cross(new_z, new_x, axisa=0, axisb=0, axisc=0)
    new_x, new_y, new_z = new_x.flatten(), new_y.flatten(), new_z.flatten()

    # if new_y is not a unit vector, new_x and new_z are not orthogonal:
    if not np.isclose(np.linalg.norm(new_y), 1):
        raise ValueError("x and z coords do not describe orthogonal positions")

    # make rotation matrix and do the rotation
    rotation = np.vstack((new_x, new_y, new_z))
    rotated_xyz = xp.asarray(rotation) @ xyz

    # convert back to spherical
    _, colatitude, rotated_longitude = cartesian_to_spherical(rotated_xyz, degrees=degrees)

    quarter_turn = 90 if degrees else np.pi / 2
    return quarter_turn - colatitude, rotated_longitude


def rotate_spherical_by_matrix(
    latitude,
    longitude,
    rotation,
    *,
    east=None,
    north=None,
    degrees=True,
):
    """Rotate spherical positions and optional east/north components.

    ``rotation`` maps source Cartesian components to target Cartesian
    components. Angles are in degrees unless ``degrees`` is false. Returned
    longitudes use the signed interval centred on zero.
    """
    if (east is None) != (north is None):
        raise ValueError("east and north must be provided together")

    xp = get_array_module(latitude, longitude, rotation, east, north)
    if east is None:
        latitude, longitude = xp.broadcast_arrays(latitude, longitude)
    else:
        latitude, longitude, east, north = xp.broadcast_arrays(latitude, longitude, east, north)
    latitude = xp.asarray(latitude)
    longitude = xp.asarray(longitude)
    rotation = xp.asarray(rotation)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")

    angle_scale = DEGREES_TO_RADIANS if degrees else 1.0
    output_scale = RADIANS_TO_DEGREES if degrees else 1.0
    latitude_radians = latitude * angle_scale
    longitude_radians = longitude * angle_scale
    cos_latitude = xp.cos(latitude_radians)
    source_cartesian = xp.stack(
        (
            cos_latitude * xp.cos(longitude_radians),
            cos_latitude * xp.sin(longitude_radians),
            xp.sin(latitude_radians),
        ),
        axis=-1,
    )
    target_cartesian = xp.einsum("ij,...j->...i", rotation, source_cartesian)
    target_latitude = output_scale * xp.arctan2(
        target_cartesian[..., 2],
        xp.hypot(target_cartesian[..., 0], target_cartesian[..., 1]),
    )
    target_longitude = output_scale * xp.arctan2(
        target_cartesian[..., 1], target_cartesian[..., 0]
    )

    if east is None:
        return target_latitude, target_longitude

    source_basis = _enu_basis(latitude, longitude, degrees=degrees)
    source_enu = xp.stack((east, north, xp.zeros_like(east)), axis=-1)
    source_vector = xp.einsum("...ij,...j->...i", source_basis, source_enu)
    target_vector = xp.einsum("ij,...j->...i", rotation, source_vector)
    target_basis = _enu_basis(target_latitude, target_longitude, degrees=degrees)
    target_enu = xp.einsum("...ij,...i->...j", target_basis, target_vector)
    return target_latitude, target_longitude, target_enu[..., 0], target_enu[..., 1]


def _enu_basis(latitude, longitude, *, degrees=True):
    """Return ENU basis vectors as columns in Cartesian coordinates."""
    xp = get_array_module(latitude, longitude)
    latitude, longitude = xp.broadcast_arrays(latitude, longitude)
    angle_scale = DEGREES_TO_RADIANS if degrees else 1.0
    latitude = xp.asarray(latitude) * angle_scale
    longitude = xp.asarray(longitude) * angle_scale

    east = xp.stack((-xp.sin(longitude), xp.cos(longitude), xp.zeros_like(longitude)), axis=-1)
    north = xp.stack(
        (
            -xp.sin(latitude) * xp.cos(longitude),
            -xp.sin(latitude) * xp.sin(longitude),
            xp.cos(latitude),
        ),
        axis=-1,
    )
    up = xp.stack(
        (
            xp.cos(latitude) * xp.cos(longitude),
            xp.cos(latitude) * xp.sin(longitude),
            xp.sin(latitude),
        ),
        axis=-1,
    )
    return xp.stack((east, north, up), axis=-1)


def enu_to_ecef(vectors, latitude, longitude):
    """Convert east, north, and up vectors to Earth-centred Cartesian components."""
    xp = get_array_module(vectors, latitude, longitude)
    vectors = xp.asarray(vectors)
    return xp.einsum("...ij,...j->...i", _enu_basis(latitude, longitude), vectors)


def ecef_to_enu(vectors, latitude, longitude):
    """Convert Earth-centred Cartesian vectors to east, north, and up components."""
    xp = get_array_module(vectors, latitude, longitude)
    vectors = xp.asarray(vectors)
    return xp.einsum("...ij,...i->...j", _enu_basis(latitude, longitude), vectors)


__all__ = [
    "cartesian_to_spherical",
    "ecef_to_enu",
    "enu_to_ecef",
    "rotate_spherical_by_matrix",
    "rotate_spherical_coordinates",
    "spherical_to_cartesian",
]
