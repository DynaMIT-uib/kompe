"""Cartesian elementary current systems (CECS).

Kompe uses ``x`` eastward, ``y`` northward, and ``z`` upward.  This
differs from Vanhamäki (2007), where ``x`` is northward, ``y`` is
eastward, and ``z`` is downward; direct comparison with that reference
therefore shows component and sign changes.

A positive curl-free amplitude represents downward field-aligned
current and radially outward sheet current.  A positive
divergence-free amplitude circulates clockwise when viewed from above
(``-phi_hat``).

References
----------
Vanhamäki, H. (2007), *Theoretical modeling of ionospheric
electrodynamics including induction effects*, Finnish Meteorological
Institute Contributions.

Vanhamäki, H., Viljanen, A. and Amm, O. (2005), *Annales Geophysicae*,
23, 1735–1746. https://doi.org/10.5194/angeo-23-1735-2005.
"""

import numpy as np

from kompe.constants import MU0
from kompe.math import get_array_module

d2r = np.pi / 180

_CURRENT_TYPES = frozenset({"curl_free", "divergence_free"})


def _validate_current_type(current_type):
    if current_type not in _CURRENT_TYPES:
        choices = ", ".join(sorted(_CURRENT_TYPES))
        raise ValueError(f"current_type must be one of: {choices}")


def relative_coordinates(x, y, z, x_poles, y_poles, z_poles):
    """Return evaluation coordinates relative to every Cartesian pole."""
    xp = get_array_module(x, y, z, x_poles, y_poles, z_poles)
    try:
        x, y, z = xp.broadcast_arrays(
            xp.asarray(x, dtype=float),
            xp.asarray(y, dtype=float),
            xp.asarray(z, dtype=float),
        )
        x_poles, y_poles, z_poles = xp.broadcast_arrays(
            xp.asarray(x_poles, dtype=float),
            xp.asarray(y_poles, dtype=float),
            xp.asarray(z_poles, dtype=float),
        )
    except ValueError as error:
        raise ValueError(
            "evaluation coordinates and pole coordinates must each be broadcast-compatible"
        ) from error
    if not all(bool(xp.isfinite(values).all()) for values in (x, y, z, x_poles, y_poles, z_poles)):
        raise ValueError("Cartesian coordinates must be finite")

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    x_c = x_poles.reshape(1, -1)
    y_c = y_poles.reshape(1, -1)
    z_c = z_poles.reshape(1, -1)

    x_prime = x - x_c
    y_prime = y - y_c
    z_prime = z - z_c

    return x_prime, y_prime, z_prime


def cartesian_distance(x, y, z, x_poles, y_poles, z_poles):
    """Return point-by-pole three-dimensional distances."""
    x_prime, y_prime, z_prime = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)

    xp = get_array_module(x_prime, y_prime, z_prime)
    dist = xp.sqrt(x_prime**2 + y_prime**2 + z_prime**2)

    return dist


def horizontal_distance(x, y, z, x_poles, y_poles, z_poles):
    """Return point-by-pole horizontal distances ``rho``."""
    x_prime, y_prime, _ = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)

    xp = get_array_module(x_prime, y_prime)
    rho = xp.sqrt(x_prime**2 + y_prime**2)

    return rho


def horizontal_azimuth(x, y, z, x_poles, y_poles, z_poles, return_degrees=False):
    """Return point-by-pole horizontal azimuths."""
    x_prime, y_prime, _ = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)

    xp = get_array_module(x_prime, y_prime)
    phi = xp.arctan2(y_prime, x_prime)

    if return_degrees:
        phi = phi / d2r

    return phi


def azimuth_unit_vectors(x, y, z, x_poles, y_poles, z_poles):
    """Return Cartesian ``phi_hat`` vectors with shape (points, poles, 3)."""
    phi_prime = horizontal_azimuth(x, y, z, x_poles, y_poles, z_poles)
    xp = get_array_module(phi_prime)

    phi_hat = xp.transpose(
        xp.stack([-xp.sin(phi_prime), xp.cos(phi_prime), xp.zeros_like(phi_prime)]),
        axes=[1, 2, 0],
    )

    return phi_hat


def radial_unit_vectors(x, y, z, x_poles, y_poles, z_poles):
    """Return Cartesian ``rho_hat`` vectors with shape (points, poles, 3)."""
    phi_prime = horizontal_azimuth(x, y, z, x_poles, y_poles, z_poles)
    xp = get_array_module(phi_prime)

    rho_hat = xp.transpose(
        xp.stack([xp.cos(phi_prime), xp.sin(phi_prime), xp.zeros_like(phi_prime)]),
        axes=[1, 2, 0],
    )

    return rho_hat


def cartesian_current_matrices(
    x,
    y,
    z,
    x_poles,
    y_poles,
    z_poles,
    current_type="divergence_free",
    constant=1.0 / (2 * np.pi),
    dz_tolerance=10.0,
):
    """Return CECS-amplitude maps for horizontal current density.

    Parameters
    ----------
    x, y, z : array-like
        Broadcast-compatible evaluation coordinates in meters.
    x_poles, y_poles, z_poles : array-like
        Broadcast-compatible CECS pole coordinates in meters.
    current_type : str, optional
        Either ``"divergence_free"`` or ``"curl_free"``.
    constant : float, optional
        Kernel scale; defaults to ``1 / (2 pi)``.
    dz_tolerance : float, optional
        Maximum vertical separation at which a pole contributes to the
        horizontal current sheet.

    Returns
    -------
    Gx, Gy : array
        Point-by-pole matrices satisfying ``jx = Gx @ amplitudes`` and
        ``jy = Gy @ amplitudes``.

    """
    _validate_current_type(current_type)
    constant = float(constant)
    dz_tolerance = float(dz_tolerance)
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    if not np.isfinite(dz_tolerance) or dz_tolerance < 0.0:
        raise ValueError("dz_tolerance must be finite and non-negative")

    _, _, z_prime = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)
    xp = get_array_module(z_prime)

    rho = horizontal_distance(x, y, z, x_poles, y_poles, z_poles)

    if current_type == "divergence_free":
        unit_vec = -azimuth_unit_vectors(x, y, z, x_poles, y_poles, z_poles)
    else:
        unit_vec = radial_unit_vectors(x, y, z, x_poles, y_poles, z_poles)

    # get the scalar part of Amm's divergence-free CECS:
    coeff = constant / rho
    Gx = coeff * unit_vec[:, :, 0]
    Gy = coeff * unit_vec[:, :, 1]

    outside_sheet = xp.abs(z_prime) > dz_tolerance
    Gx = xp.where(outside_sheet, 0.0, Gx)
    Gy = xp.where(outside_sheet, 0.0, Gy)
    return Gx, Gy


def cartesian_magnetic_field_matrices(
    x,
    y,
    z,
    x_poles,
    y_poles,
    z_poles,
    current_type="divergence_free",
    constant=MU0 / (4.0 * np.pi),
):
    """Return CECS-amplitude maps for the Cartesian magnetic field.

    Based on equations (A.10) and (A.11) of Vanhamäki (2007).

    Parameters
    ----------
    x, y, z : array-like
        Broadcast-compatible evaluation coordinates in meters.
    x_poles, y_poles, z_poles : array-like
        Broadcast-compatible CECS pole coordinates in meters.
    current_type : str, optional
        Either ``"divergence_free"`` or ``"curl_free"``.
    constant : float, optional
        Kernel scale; defaults to ``mu0 / (4 pi)``.

    Returns
    -------
    Gx, Gy, Gz : array
        Point-by-pole matrices satisfying ``Bi = Gi @ amplitudes``.

    """
    _validate_current_type(current_type)
    constant = float(constant)
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    _, _, z_prime = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)
    xp = get_array_module(z_prime)

    rho = horizontal_distance(x, y, z, x_poles, y_poles, z_poles)

    coeff = constant / rho

    # G matrix scale factors
    if current_type == "divergence_free":
        rho_hat = radial_unit_vectors(x, y, z, x_poles, y_poles, z_poles)

        G_rho_prime = (
            -coeff * (1 - xp.abs(z_prime) / xp.sqrt(rho**2 + z_prime**2)) * xp.sign(z_prime)
        )

        Gx = G_rho_prime * rho_hat[:, :, 0]
        Gy = G_rho_prime * rho_hat[:, :, 1]

        Gz = -coeff * rho / xp.sqrt(rho**2 + z_prime**2)

    elif current_type == "curl_free":
        phi_hat = azimuth_unit_vectors(x, y, z, x_poles, y_poles, z_poles)

        heaviside = xp.where((z_prime > 0.0) | xp.isclose(z_prime, 0.0), 1.0, 0.0)

        G_phi_prime = -2.0 * coeff * heaviside

        Gx = G_phi_prime * phi_hat[:, :, 0]
        Gy = G_phi_prime * phi_hat[:, :, 1]
        Gz = xp.zeros_like(phi_hat[:, :, 2])

    return Gx, Gy, Gz


__all__ = [
    "azimuth_unit_vectors",
    "cartesian_current_matrices",
    "cartesian_distance",
    "cartesian_magnetic_field_matrices",
    "horizontal_azimuth",
    "horizontal_distance",
    "radial_unit_vectors",
    "relative_coordinates",
]
