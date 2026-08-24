"""Canonical spherical elementary-current-system kernels."""

import numpy as np

from kompe.constants import EARTH_RADIUS_M, MU0
from kompe.math.backend import get_array_module
from kompe.spherical import ecef_to_enu

DEGREES_TO_RADIANS = np.pi / 180


def _clip_dot_product(x):
    """Keep spherical dot products inside the roundoff-safe cosine range."""
    xp = get_array_module(x)
    return xp.clip(x, -1.0, 1.0)


def _unit_ecef_vectors(xp, latitude, longitude):
    """Return unit ECEF position vectors for geographic coordinates."""
    latitude = xp.asarray(latitude).reshape(-1) * DEGREES_TO_RADIANS
    longitude = xp.asarray(longitude).reshape(-1) * DEGREES_TO_RADIANS
    position = xp.stack(
        (
            xp.cos(latitude) * xp.cos(longitude),
            xp.cos(latitude) * xp.sin(longitude),
            xp.sin(latitude),
        ),
        axis=1,
    )
    return position


def _spherical_secs_geometry(lat, lon, pole_latitudes, pole_longitudes):
    """Return angular distance and poleward directions for SECS kernels."""
    xp = get_array_module(lat, lon, pole_latitudes, pole_longitudes)
    evaluation_position = _unit_ecef_vectors(xp, lat, lon)
    pole_position = _unit_ecef_vectors(xp, pole_latitudes, pole_longitudes)

    # Unit tangent from each evaluation point towards each pole (N x P x 3).
    poleward_ecef = pole_position[None, :, :] - evaluation_position[:, None, :]
    poleward_ecef -= (
        xp.einsum("npi,ni->np", poleward_ecef, evaluation_position)[..., None]
        * evaluation_position[:, None, :]
    )
    poleward_norm = xp.linalg.norm(poleward_ecef, axis=-1)[..., None]
    if xp is np:
        with np.errstate(invalid="ignore", divide="ignore"):
            poleward_ecef = poleward_ecef / poleward_norm
    else:
        poleward_ecef = poleward_ecef / poleward_norm

    poleward_enu = ecef_to_enu(
        poleward_ecef,
        xp.asarray(lat).reshape(-1, 1),
        xp.asarray(lon).reshape(-1, 1),
    )[..., :2]
    theta = xp.arccos(
        _clip_dot_product(xp.einsum("ni,pi->np", evaluation_position, pole_position))
    )
    return xp, evaluation_position, poleward_enu, theta


def angular_distance(lat, lon, pole_latitudes, pole_longitudes, return_degrees=False):
    """Return angular distances from evaluation points to SECS poles.

    Coordinates are geographic degrees. The result has shape
    ``(number of evaluation points, number of poles)`` and is in radians
    unless ``return_degrees`` is true.
    """

    xp = get_array_module(lat, lon, pole_latitudes, pole_longitudes)
    evaluation_position = _unit_ecef_vectors(xp, lat, lon)
    pole_position = _unit_ecef_vectors(xp, pole_latitudes, pole_longitudes)
    theta = xp.arccos(
        _clip_dot_product(xp.einsum("ni,pi->np", evaluation_position, pole_position))
    )

    if return_degrees:
        theta = theta / DEGREES_TO_RADIANS

    return theta


def scalar_green_matrix(
    lat,
    lon,
    pole_latitudes,
    pole_longitudes,
    *,
    quantity,
    normalization=1.0 / (4 * np.pi),
):
    """Return a scalar SECS Green matrix.

    ``quantity="potential"`` returns the potential whose negative surface
    gradient gives curl-free SECS current. ``quantity="current_magnitude"``
    returns the scalar ``cot(theta/2)`` profile shared by the horizontal
    current kernels before direction and radius scaling are applied.
    """
    theta = angular_distance(lat, lon, pole_latitudes, pole_longitudes)
    xp = get_array_module(theta)

    if quantity == "potential":
        if xp is np:
            with np.errstate(divide="ignore"):
                return -2 * normalization * xp.log(xp.sin(theta / 2))
        return -2 * normalization * xp.log(xp.sin(theta / 2))
    if quantity == "current_magnitude":
        return normalization / xp.tan(theta / 2)
    raise ValueError('quantity must be "potential" or "current_magnitude"')


def surface_current_matrices(
    lat,
    lon,
    pole_latitudes,
    pole_longitudes,
    current_type="divergence_free",
    normalization=1.0 / (4 * np.pi),
    source_radius=EARTH_RADIUS_M + 110 * 1e3,
    singularity_limit=0,
):
    """Return matrices mapping SECS amplitudes to horizontal current density.

    The result is ``(east, north)``. For both current modes the magnitude away
    from the singularity is

    ``normalization / source_radius * cot(theta / 2)``.

    Curl-free currents point away from each pole; divergence-free currents are
    the corresponding clockwise rotation. Coordinates are geographic degrees,
    radii and ``singularity_limit`` must use the same length unit, and each
    matrix has shape ``(number of evaluation points, number of poles)``.

    Parameters
    ----------
    current_type : {"curl_free", "divergence_free"}
        Physical orientation of the elementary currents.
    normalization : float, optional
        Multiplicative Green-function normalization; default ``1/(4*pi)``.
    source_radius : float, optional
        Radius of the current sheet.
    singularity_limit : float, optional
        Regularization distance around each pole. Zero retains the singular
        kernel. Positive values use equations 2.43--2.44 of Vanhamäki and
        Juusola (2020).
    """

    xp, _, poleward_enu, theta = _spherical_secs_geometry(
        lat, lon, pole_latitudes, pole_longitudes
    )

    if current_type == "divergence_free":
        # Rotate the poleward tangent clockwise in the local horizontal plane.
        current_direction = xp.dstack(
            (poleward_enu[:, :, 1], -poleward_enu[:, :, 0])
        )  # north -> east and east -> south
    elif current_type == "curl_free":
        current_direction = -poleward_enu  # outward from SECS
    else:
        raise ValueError('current_type must be "divergence_free" or "curl_free"')

    current_magnitude = normalization / xp.tan(theta / 2) / source_radius

    # Equations 2.43--2.44 in Vanhamäki and Juusola (2020).
    theta0 = singularity_limit / source_radius
    if theta0 > 0:
        alpha = 1 / np.tan(theta0 / 2) ** 2
        regularized = normalization * alpha * xp.tan(theta / 2) / source_radius
        current_magnitude = xp.where(theta < theta0, regularized, current_magnitude)

    east = current_magnitude * current_direction[:, :, 0]
    north = current_magnitude * current_direction[:, :, 1]
    return east, north


def magnetic_field_matrices(
    lat,
    lon,
    r,
    pole_latitudes,
    pole_longitudes,
    current_type="divergence_free",
    normalization=1.0 / (4 * np.pi),
    source_radius=EARTH_RADIUS_M + 110 * 1e3,
    singularity_limit=0,
    induction_nullification_radius=None,
):
    """Return matrices mapping SECS amplitudes to magnetic field.

    The result is ``(east, north, radial)`` and each matrix has shape
    ``(number of evaluation points, number of poles)``. Coordinates are
    geographic degrees; ``r``, ``source_radius``, ``singularity_limit``, and
    ``induction_nullification_radius`` must use the same length unit.

    The field follows equations 9--10 of Amm and Viljanen (1999), equivalently
    equations 2.13--2.14 of Vanhamäki and Juusola (2020). A positive
    ``singularity_limit`` regularizes only the curl-free field, following
    section 2.10.2 and equation 2.46 of the latter reference.


    Parameters
    ----------
    r : array-like
        Scalar evaluation radius or one radius per evaluation point.
    current_type : {"curl_free", "divergence_free"}
        Physical current-system mode.
    normalization : float, optional
        Multiplicative Green-function normalization; default ``1/(4*pi)``.
    source_radius : float, optional
        Radius of the current sheet.
    singularity_limit : float, optional
        Curl-free regularization distance around each pole.
    induction_nullification_radius : float or None, optional
        Radius at which the divergence-free primary and telluric image
        currents have cancelling radial magnetic field. The image-current
        construction follows appendix A of Juusola et al. (2016).
    """

    xp = get_array_module(lat, lon, r, pole_latitudes, pole_longitudes)
    xp, evaluation_position, poleward_enu, theta = _spherical_secs_geometry(
        xp.asarray(lat),
        xp.asarray(lon),
        xp.asarray(pole_latitudes),
        xp.asarray(pole_longitudes),
    )

    evaluation_radius = xp.asarray(r)
    if evaluation_radius.size == 1:
        evaluation_radius = xp.broadcast_to(
            evaluation_radius.reshape(1, 1), (evaluation_position.shape[0], 1)
        )
    else:
        evaluation_radius = evaluation_radius.flatten()[:, None]

    below_current_sheet = evaluation_radius.flatten() <= source_radius

    # G matrix scale factors
    if current_type == "divergence_free":
        s = xp.minimum(evaluation_radius, source_radius) / xp.maximum(
            evaluation_radius, source_radius
        )
        root = xp.sqrt(1 + s**2 - 2 * s * xp.cos(theta))

        Ar = MU0 * normalization / evaluation_radius  # common factor radial direction
        Sr = xp.where(below_current_sheet[:, None], 1 / root - 1, s / root - s)
        Gr = Ar * Sr

        An_ = (
            MU0 * normalization / (evaluation_radius * xp.sin(theta))
        )  # common factor local northward (note sign difference wrt theta) direction
        cos_theta = xp.cos(theta)
        Sn_ = xp.where(
            below_current_sheet[:, None],
            (s - cos_theta) / root + cos_theta,
            (1 - s * cos_theta) / root - 1,
        )
        Gn_ = An_ * Sn_

        # calculate geo east, north:
        Ge = Gn_ * poleward_enu[:, :, 0]
        Gn = Gn_ * poleward_enu[:, :, 1]

    elif current_type == "curl_free":
        # G matrix for local eastward component
        Ge_ = -MU0 * normalization / xp.tan(theta / 2) / evaluation_radius

        # apply modifications to handle singularities:
        theta0 = singularity_limit / source_radius
        if theta0 > 0:
            alpha = 1 / np.tan(theta0 / 2) ** 2
            regularized = -MU0 * normalization * alpha * xp.tan(theta / 2) / evaluation_radius
            Ge_ = xp.where(theta < theta0, regularized, Ge_)

        # zero below current sheet:
        Ge_ = Ge_ * (~below_current_sheet[:, None])

        # calculate geo east, north, radial:
        Ge = (
            Ge_ * poleward_enu[:, :, 1]
        )  # eastward component of poleward_enu is northward in the local azimuthal direction
        Gn = -Ge_ * poleward_enu[:, :, 0]
        Gr = Ge_ * 0  # no radial component

    else:
        raise ValueError('current_type must be "divergence_free" or "curl_free"')

    if induction_nullification_radius is not None and current_type == "divergence_free":
        # include the effect of telluric image currents
        radius = induction_nullification_radius**2 / source_radius
        amplitude_factor = -source_radius / induction_nullification_radius

        Ge_, Gn_, Gr_ = magnetic_field_matrices(
            lat,
            lon,
            evaluation_radius,
            pole_latitudes,
            pole_longitudes,
            current_type="divergence_free",
            normalization=normalization,
            source_radius=radius,
        )
        Ge = Ge + amplitude_factor * Ge_
        Gn = Gn + amplitude_factor * Gn_
        Gr = Gr + amplitude_factor * Gr_

    return Ge, Gn, Gr


__all__ = [
    "angular_distance",
    "magnetic_field_matrices",
    "scalar_green_matrix",
    "surface_current_matrices",
]
