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


def enu_to_ecef(vectors, latitude, longitude):
    """Convert east, north, and up vectors to Earth-centred Cartesian components."""
    xp = get_array_module(vectors, latitude, longitude)
    vectors = xp.asarray(vectors)
    latitude = xp.asarray(latitude)
    longitude = xp.asarray(longitude)

    # construct unit vectors in east, north, up directions:
    phi = longitude * DEGREES_TO_RADIANS
    theta = (90 - latitude) * DEGREES_TO_RADIANS

    east = xp.vstack((-xp.sin(phi), xp.cos(phi), xp.zeros_like(phi))).T
    north = xp.vstack(
        (-xp.cos(theta) * xp.cos(phi), -xp.cos(theta) * xp.sin(phi), xp.sin(theta))
    ).T
    up = xp.vstack((xp.sin(theta) * xp.cos(phi), xp.sin(theta) * xp.sin(phi), xp.cos(theta))).T

    # ENU basis vectors form the columns of the rotation matrix.
    enu_basis = xp.stack((east, north, up), axis=2)  # (N, 3, 3)

    # perform the rotations:
    return xp.einsum("nij,nj->ni", enu_basis, vectors)


def ecef_to_enu(vectors, latitude, longitude):
    """Convert Earth-centred Cartesian vectors to east, north, and up components."""
    xp = get_array_module(vectors, latitude, longitude)
    vectors = xp.asarray(vectors)
    latitude = xp.asarray(latitude)
    longitude = xp.asarray(longitude)

    phi = longitude * DEGREES_TO_RADIANS
    theta = (90 - latitude) * DEGREES_TO_RADIANS
    east = xp.vstack((-xp.sin(phi), xp.cos(phi), xp.zeros_like(phi))).T
    north = xp.vstack(
        (-xp.cos(theta) * xp.cos(phi), -xp.cos(theta) * xp.sin(phi), xp.sin(theta))
    ).T
    up = xp.vstack((xp.sin(theta) * xp.cos(phi), xp.sin(theta) * xp.sin(phi), xp.cos(theta))).T
    return xp.column_stack(
        (
            xp.einsum("ni,ni->n", vectors, east),
            xp.einsum("ni,ni->n", vectors, north),
            xp.einsum("ni,ni->n", vectors, up),
        )
    )


__all__ = [
    "cartesian_to_spherical",
    "ecef_to_enu",
    "enu_to_ecef",
    "rotate_spherical_coordinates",
    "spherical_to_cartesian",
]
