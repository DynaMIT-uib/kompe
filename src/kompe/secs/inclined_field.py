"""Inclined field-aligned current corrections for spherical SECS."""

import numpy as np

from kompe.constants import EARTH_RADIUS_M, MU0
from kompe.math.backend import get_array_module
from kompe.secs.kernels import magnetic_field_matrices
from kompe.spherical_coordinates import enu_to_ecef


def _semi_infinite_current_magnetic_field_matrices(
    lat,
    lon,
    radius,
    connection_latitudes,
    connection_longitudes,
    connection_radii,
    direction_east,
    direction_north,
    direction_radial,
):
    """Return magnetic-field matrices for semi-infinite straight currents.

    The direction components describe each current line in the local east,
    north, and radial frame at its finite endpoint.
    """
    xp = get_array_module(
        lat,
        lon,
        radius,
        connection_latitudes,
        connection_longitudes,
        connection_radii,
        direction_east,
        direction_north,
        direction_radial,
    )

    # Broadcast evaluation points independently of the current-wedge parameters.
    evaluation_coordinates = xp.broadcast_arrays(
        *(xp.asarray(value) for value in (lat, lon, radius))
    )
    wedge_parameters = xp.broadcast_arrays(
        *(
            xp.asarray(value)
            for value in (
                connection_latitudes,
                connection_longitudes,
                connection_radii,
                direction_east,
                direction_north,
                direction_radial,
            )
        )
    )

    evaluation_coordinates = [value.reshape(-1) for value in evaluation_coordinates]
    wedge_parameters = [value.reshape(-1) for value in wedge_parameters]

    n_points = evaluation_coordinates[0].size
    n_wedges = wedge_parameters[0].size

    # Construct 3 x N array, r_ecef, of N vectors pointing at evaluation points (ECEF)
    phi = xp.deg2rad(evaluation_coordinates[1])
    theta = xp.deg2rad(90 - evaluation_coordinates[0])
    evaluation_radius = evaluation_coordinates[2]
    evaluation_ecef = evaluation_radius * xp.vstack(
        (xp.cos(phi) * xp.sin(theta), xp.sin(phi) * xp.sin(theta), xp.cos(theta))
    )

    # Construct 3 x K array, s_ecef, of K vectors pointing at wedge intersection (ECEF)
    connection_phi, connection_theta, connection_radius = (
        xp.deg2rad(wedge_parameters[1]),
        xp.deg2rad(90 - wedge_parameters[0]),
        wedge_parameters[2],
    )
    connection_ecef = connection_radius * xp.vstack(
        (
            xp.cos(connection_phi) * xp.sin(connection_theta),
            xp.sin(connection_phi) * xp.sin(connection_theta),
            xp.cos(connection_theta),
        )
    )

    # Unit vectors pointing upward along the inclined wedge legs in ECEF.
    direction_enu = xp.vstack((wedge_parameters[3], wedge_parameters[4], wedge_parameters[5]))
    direction_enu = direction_enu / xp.linalg.norm(direction_enu, axis=0)
    upward_sign = xp.sign(direction_enu[2])
    direction_enu = direction_enu * upward_sign.reshape((1, -1))
    direction_ecef = enu_to_ecef(direction_enu.T, wedge_parameters[0], wedge_parameters[1]).T

    # Find the lengths along j (from s) that are closest to evaluation points (N x K)
    distance_along = xp.einsum("in,ik->nk", evaluation_ecef, direction_ecef) - xp.sum(
        connection_ecef * direction_ecef, axis=0
    ).reshape((1, n_wedges))

    # Find vectors pointing at evaluation points from the closest point along j (3 x N x K):
    perpendicular_x = (
        evaluation_ecef[0].reshape((n_points, 1))
        - connection_ecef[0].reshape((1, n_wedges))
        - direction_ecef[0] * distance_along
    )
    perpendicular_y = (
        evaluation_ecef[1].reshape((n_points, 1))
        - connection_ecef[1].reshape((1, n_wedges))
        - direction_ecef[1] * distance_along
    )
    perpendicular_z = (
        evaluation_ecef[2].reshape((n_points, 1))
        - connection_ecef[2].reshape((1, n_wedges))
        - direction_ecef[2] * distance_along
    )
    perpendicular_ecef = xp.stack((perpendicular_x, perpendicular_y, perpendicular_z))

    # Distances from evaluation points to the closest points on each line (N x K).
    perpendicular_distance = xp.sqrt(perpendicular_x**2 + perpendicular_y**2 + perpendicular_z**2)

    # normalized versions of p_ecef vectors:
    perpendicular_unit = perpendicular_ecef / perpendicular_distance.reshape(
        (1, n_points, n_wedges)
    )

    # Biot-Savart direction: perpendicular-to-wire crossed with wire direction.
    field_direction_x = (
        perpendicular_unit[1] * direction_ecef[2] - perpendicular_unit[2] * direction_ecef[1]
    )
    field_direction_y = (
        perpendicular_unit[2] * direction_ecef[0] - perpendicular_unit[0] * direction_ecef[2]
    )
    field_direction_z = (
        perpendicular_unit[0] * direction_ecef[1] - perpendicular_unit[1] * direction_ecef[0]
    )
    field_direction = xp.stack((field_direction_x, field_direction_y, field_direction_z))

    # Angle from the perpendicular point on each wire to its finite endpoint.
    endpoint_angle = xp.arctan(-distance_along / perpendicular_distance)

    # magnetic field scaling factor (N x K):
    field_scale = MU0 / (4 * np.pi * perpendicular_distance) * (1 - xp.sin(endpoint_angle))
    field_scale = field_scale.reshape((1, n_points, n_wedges))

    # (3 x N x K) array that map current magnitudes to ECEF components of the magnetic field:
    field_ecef = field_scale * field_direction

    # convert GB_ecef to enu - three matrices that are N x K:
    phi, theta = phi.reshape((n_points, 1)), theta.reshape((n_points, 1))
    G_e = -xp.sin(phi) * field_ecef[0] + xp.cos(phi) * field_ecef[1]
    G_n = (
        -xp.cos(theta) * xp.cos(phi) * field_ecef[0]
        - xp.cos(theta) * xp.sin(phi) * field_ecef[1]
        + xp.sin(theta) * field_ecef[2]
    )
    G_r = (
        xp.sin(theta) * xp.cos(phi) * field_ecef[0]
        + xp.sin(theta) * xp.sin(phi) * field_ecef[1]
        + xp.cos(theta) * field_ecef[2]
    )

    return G_e, G_n, G_r


def current_wedge_magnetic_field_matrices(
    lat,
    lon,
    radius,
    connection_latitudes,
    connection_longitudes,
    connection_radii,
    direction_east,
    direction_north,
    direction_radial,
):
    """Return magnetic-field matrices for semi-infinite current wedges.

    Each wedge joins an inclined semi-infinite current to a radial
    semi-infinite current at one connection point. The direction components
    describe the inclined leg in the local east, north, and radial frame. The
    returned matrices map wedge currents in amperes to east, north, and radial
    magnetic-field components.

    This straight-line field-line approximation is intended as a first-order
    mid-latitude correction.
    """
    inclined_leg = _semi_infinite_current_magnetic_field_matrices(
        lat,
        lon,
        radius,
        connection_latitudes,
        connection_longitudes,
        connection_radii,
        direction_east,
        direction_north,
        direction_radial,
    )
    xp = get_array_module(direction_east, direction_north, direction_radial)
    radial_leg = _semi_infinite_current_magnetic_field_matrices(
        lat,
        lon,
        radius,
        connection_latitudes,
        connection_longitudes,
        connection_radii,
        xp.zeros_like(direction_east),
        xp.zeros_like(direction_north),
        direction_radial,
    )
    return tuple(
        inclined_component - radial_component
        for inclined_component, radial_component in zip(inclined_leg, radial_leg, strict=True)
    )


def inclined_secs_magnetic_field_matrices(
    lat,
    lon,
    radius,
    pole_latitudes,
    pole_longitudes,
    main_field_east,
    main_field_north,
    main_field_radial,
    *,
    source_radius=EARTH_RADIUS_M + 110e3,
):
    """Return curl-free SECS magnetic matrices for inclined field lines.

    The standard radial-field-line SECS field is combined with the current-wedge
    correction defined by the background magnetic-field direction at each pole.
    This straight-line correction is intended for mid-latitude field lines.
    """
    xp = get_array_module(
        lat,
        lon,
        radius,
        pole_latitudes,
        pole_longitudes,
        main_field_east,
        main_field_north,
        main_field_radial,
    )
    pole_latitudes = xp.asarray(pole_latitudes)
    source_radii = xp.broadcast_to(xp.asarray(source_radius), pole_latitudes.shape)

    wedge_east, wedge_north, wedge_radial = current_wedge_magnetic_field_matrices(
        lat,
        lon,
        radius,
        pole_latitudes,
        pole_longitudes,
        source_radii,
        main_field_east,
        main_field_north,
        main_field_radial,
    )
    radial_east, radial_north, radial_radial = magnetic_field_matrices(
        lat,
        lon,
        radius,
        pole_latitudes,
        pole_longitudes,
        source_radius=source_radius,
        current_type="curl_free",
    )

    return (
        radial_east + wedge_east,
        radial_north + wedge_north,
        radial_radial + wedge_radial,
    )


__all__ = [
    "current_wedge_magnetic_field_matrices",
    "inclined_secs_magnetic_field_matrices",
]
