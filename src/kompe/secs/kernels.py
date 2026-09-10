"""Canonical spherical elementary-current-system kernels."""

import numpy as np

from kompe.constants import EARTH_RADIUS_M, MU0
from kompe.math.backend import get_array_module
from kompe.spherical_coordinates import ecef_to_enu

DEGREES_TO_RADIANS = np.pi / 180


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
    # The tangent length is sin(theta); atan2 retains small separations
    # that arccos(dot) would round to zero. Reuse the geometry already built.
    theta = xp.arctan2(
        poleward_norm[..., 0], xp.einsum("ni,pi->np", evaluation_position, pole_position)
    )
    # At coincidence the direction is undefined. Keep its zero tangent so
    # regularized currents and off-sheet magnetic fields have their exact
    # zero horizontal limit; an infinite unregularized kernel stays singular.
    poleward_ecef = poleward_ecef / xp.where(poleward_norm == 0, 1.0, poleward_norm)

    poleward_enu = ecef_to_enu(
        poleward_ecef,
        xp.asarray(lat).reshape(-1, 1),
        xp.asarray(lon).reshape(-1, 1),
    )[..., :2]
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
    # Half-chords resolve both coincident and antipodal limits without
    # constructing the tangent directions needed by the vector kernels.
    difference = pole_position[None, :, :] - evaluation_position[:, None, :]
    total = pole_position[None, :, :] + evaluation_position[:, None, :]
    theta = 2 * xp.arctan2(xp.linalg.norm(difference, axis=-1), xp.linalg.norm(total, axis=-1))

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

    ``quantity="curl_free_potential"`` returns Phi with J_cf = -grad_s(Phi).
    ``quantity="divergence_free_potential"`` returns Psi with
    J_df = rhat x grad_s(Psi), using Kompe's shared Helmholtz signs.
    ``quantity="current_profile"`` returns the cot(theta/2) profile before
    direction and 1/radius scaling. A sum of these profiles is not the
    magnitude of the summed vector current.
    """
    theta = angular_distance(lat, lon, pole_latitudes, pole_longitudes)
    xp = get_array_module(theta)

    if quantity in {"curl_free_potential", "divergence_free_potential"}:
        sign = -1 if quantity == "curl_free_potential" else 1
        if xp is np:
            with np.errstate(divide="ignore"):
                return sign * 2 * normalization * xp.log(xp.sin(theta / 2))
        return sign * 2 * normalization * xp.log(xp.sin(theta / 2))
    if quantity == "current_profile":
        return normalization / xp.tan(theta / 2)
    raise ValueError(
        'quantity must be "curl_free_potential", "divergence_free_potential", or "current_profile"'
    )


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
        sin_half_squared = xp.sin(theta / 2) ** 2
        root = xp.sqrt((1 - s) ** 2 + 4 * s * sin_half_squared)

        Ar = MU0 * normalization / evaluation_radius  # common factor radial direction
        # Rationalized forms of equations 2.13--2.14 retain the small-angle
        # and small-radius-ratio limits without subtracting nearly equal terms.
        cos_theta = xp.cos(theta)
        Sr = s * (2 * cos_theta - s) / (root * (1 + root))
        Sr = xp.where(below_current_sheet[:, None], Sr, s * Sr)
        Gr = Ar * Sr

        # Positive here means poleward, opposite to the local theta direction.
        denominator = root * ((1 - s) + 2 * s * sin_half_squared + root)
        Gn_ = (
            Ar
            * xp.sin(theta)
            / denominator
            * xp.where(below_current_sheet[:, None], s * (1 + root), -(s**2))
        )

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
        Ge_ = xp.where(below_current_sheet[:, None], 0.0, Ge_)

        # calculate geo east, north, radial:
        Ge = (
            Ge_ * poleward_enu[:, :, 1]
        )  # eastward component of poleward_enu is northward in the local azimuthal direction
        Gn = -Ge_ * poleward_enu[:, :, 0]
        Gr = xp.zeros_like(Ge_)  # no radial component, even on the singular axis

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
