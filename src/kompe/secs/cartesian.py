"""CECS utils

NOTE: Here we use a Cartesian coordinate system x,y,z such that z is UPWARD
(and, say, x is east and y is north). This is DIFFERENT from the Cartesian
coordinate system that Vanhamäki (2007) and Vanhamäki, Viljanen, and Amm (2005)
use to define CECS. For them, x is north, y is east, and z is DOWN.

Why does it matter? Because if you compare expressions for J and B in this
implementation with their expressions, you will notice sign differences.

Also, some conventions (established by references, not me):
•A positive-amplitude CECS CF system corresponds to a DOWNWARD field-aligned
 current and a radially OUTWARD current sheet.
•A positive-amplitude CECS DF system corresponds to a current that circulates in
 the CLOCKWISE direction when seen from above (i.e., the -phihat direction).


REFERENCES
==========

Vanhamäki, H. (2007) ‘Theoretical modeling of ionospheric electrodynamics
including induction effects’, Finnish Meteorological Institute Contributions.

Vanhamäki, H., Viljanen, A. and Amm, O. (2005) ‘Induction effects on ionospheric
electric and magnetic fields’, Annales Geophysicae, 23(5), pp. 1735–1746. doi:
10.5194/angeo-23-1735-2005.

Spencer M. Hatch
October 2022
"""

import numpy as np

from kompe.constants import MU0

d2r = np.pi / 180

_CURRENT_TYPES = frozenset({"curl_free", "divergence_free"})


def _validate_current_type(current_type):
    if current_type not in _CURRENT_TYPES:
        choices = ", ".join(sorted(_CURRENT_TYPES))
        raise ValueError(f"current_type must be one of: {choices}")


def relative_coordinates(x, y, z, x_poles, y_poles, z_poles):
    """Return evaluation coordinates relative to every Cartesian pole."""
    try:
        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        x_poles, y_poles, z_poles = np.broadcast_arrays(
            np.asarray(x_poles, dtype=float),
            np.asarray(y_poles, dtype=float),
            np.asarray(z_poles, dtype=float),
        )
    except ValueError as error:
        raise ValueError(
            "evaluation coordinates and pole coordinates must each be broadcast-compatible"
        ) from error
    if not all(np.isfinite(values).all() for values in (x, y, z, x_poles, y_poles, z_poles)):
        raise ValueError("Cartesian coordinates must be finite")

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    x_c = x_poles.reshape(1, -1)
    y_c = y_poles.reshape(1, -1)
    z_c = z_poles.reshape(1, -1)

    xp = x - x_c
    yp = y - y_c
    zp = z - z_c

    return xp, yp, zp


def cartesian_distance(x, y, z, x_poles, y_poles, z_poles):
    """ " calculate distance between data point and cecs node.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    return_degrees: bool, optional
        Set to True if you want output in degrees. Default is False (radians)

    Returns
    -------
    dist: 2D array (x.size, x_cecs.size)
        Array of distances between the points
        described by (x, y, z) and the points described by
        (x_cecs, y_cecs, z_cecs).
    """

    # reshape
    xp, yp, zp = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)

    dist = np.sqrt(xp**2 + yp**2 + zp**2)

    return dist


def horizontal_distance(x, y, z, x_poles, y_poles, z_poles):
    """ " calculate rho (horizontal distance) between data points and cecs nodes.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size

    Returns
    -------
    rho: 2D array (x.size, x_cecs.size)
        Array of horizontal distances between data points
        described by (x, y, z) and cecs nodes at (x_cecs, y_cecs, z_cecs).
    """

    xp, yp, _ = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)

    rho = np.sqrt(xp**2 + yp**2)

    return rho


def horizontal_azimuth(x, y, z, x_poles, y_poles, z_poles, return_degrees=False):
    """ " calculate polar (horizontal) angle of data point relative to cecs node.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    return_degrees: bool, optional
        Set to True if you want output in degrees. Default is False (radians)

    Returns
    -------
    phi: 2D array (x.size, x_cecs.size)
        Array of polar (horizontal) angles of data points
        described by (x, y, z) relative the points described by
        (x_cecs, y_cecs, z_cecs). Unit in radians unless return_degrees is set
        to True
    """

    xp, yp, _ = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)

    phi = np.arctan2(yp, xp)

    if return_degrees:
        phi = phi / d2r

    return phi


def azimuth_unit_vectors(x, y, z, x_poles, y_poles, z_poles):
    """ " calculate polar (horizontal) unit vector for data point relative to cecs node.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size

    Returns
    -------
    phihat: 3D array (x.size, x_cecs.size, 3)
        Array of polar (horizontal) unit vectors in Cartesian coordinates of data points
        described by (x, y, z) relative the points described by
        (x_cecs, y_cecs, z_cecs).
    """

    phiprime = horizontal_azimuth(x, y, z, x_poles, y_poles, z_poles)

    phihat = np.transpose(
        np.stack([-np.sin(phiprime), np.cos(phiprime), np.zeros_like(phiprime)]), axes=[1, 2, 0]
    )

    return phihat


def radial_unit_vectors(x, y, z, x_poles, y_poles, z_poles):
    """ " calculate rho unit vector from cecs node to data point.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    lon_cecs: array-like
        Array of CECS pole longitudes [deg]
        Flattened array must havef same size as lat_cecs
        Output will be a 2D array with shape (mlat.size, mlat_cecs.size)

    Returns
    -------
    rhohat: 3D array (3, x.size, x_cecs.size)
        Array of rho unit vectors in Cartesian coordinates pointing from
        cecs nodes to data points.
    """

    phiprime = horizontal_azimuth(x, y, z, x_poles, y_poles, z_poles)

    rhohat = np.transpose(
        np.stack([np.cos(phiprime), np.sin(phiprime), np.zeros_like(phiprime)]), axes=[1, 2, 0]
    )

    return rhohat


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
    """Return matrices mapping CECS amplitudes to horizontal current density.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    x_poles, y_poles, z_poles: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    current_type: string, optional
        Either ``"divergence_free"`` or ``"curl_free"``.
    constant: float, optional
        The CECS functions are scaled by the factor 1/(2pi), which is
        the default value of 'constant'. Change if you want something
        different.
    dz_tolerance: float, optional
        Maximum vertical separation at which a pole contributes to the
        horizontal current sheet.

    Returns
    -------
    Gx: 2D array
        2D array with shape (x.size, x_cecs.size), relating CECS amplitudes
        m to the x-direction current densities at (x, y, z) via 'jx = Gx.dot(m)'
    Gy: 2D array
        2D array with shape (x.size, y_cecs.size), relating CECS amplitudes
        m to the y-direction current densities at (x, y, z) via 'jy = Gy.dot(m)'
    """

    _validate_current_type(current_type)
    constant = float(constant)
    dz_tolerance = float(dz_tolerance)
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    if not np.isfinite(dz_tolerance) or dz_tolerance < 0.0:
        raise ValueError("dz_tolerance must be finite and non-negative")

    _, _, zp = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)

    rho = horizontal_distance(x, y, z, x_poles, y_poles, z_poles)

    if current_type == "divergence_free":
        unit_vec = -azimuth_unit_vectors(x, y, z, x_poles, y_poles, z_poles)
    else:
        unit_vec = radial_unit_vectors(x, y, z, x_poles, y_poles, z_poles)

    # get the scalar part of Amm's divergence-free CECS:
    coeff = constant / rho
    Gx = coeff * unit_vec[:, :, 0]
    Gy = coeff * unit_vec[:, :, 1]

    outside_sheet = np.abs(zp) > dz_tolerance
    Gx[outside_sheet] = 0.0
    Gy[outside_sheet] = 0.0
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
    """Return matrices mapping CECS amplitudes to Cartesian magnetic field.

    Based on equations (A.10) and (A.11) of Vanhamäki (2007).

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    x_poles, y_poles, z_poles: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    current_type: string, optional
        The type of CECS function. This must be either
        'divergence_free' (default): divergence-free basis functions
        'curl_free': curl-free basis functions
    constant: float, optional
        The B^cf CECS function is scaled by the factor MU0/(2pi), while
        B^df CECS function is scaled by MU0/(4pi)

    Returns
    -------
    Gx: 2D array
        2D array with shape (x.size, x_cecs.size), relating CECS amplitudes
        m to the x-component magnetic field at (x, y, z) via 'Bx = Gx.dot(m)'
    Gy: 2D array
        2D array with shape (y.size, y_cecs.size), relating CECS amplitudes
        m to the y-component magnetic field at (x, y, z) via 'By = Gy.dot(m)'
    Gz: 2D array
        2D array with shape (z.size, z_cecs.size), relating CECS amplitudes
        m to the z-component magnetic field at (x, y, z) via 'Bz = Gz.dot(m)'


    Notes
    ------
    Variables with 'p' in name signify a quantity in the CECS node's local (i.e., "p"rimed) coordinate system

    2022/10/15 SMH
    """

    _validate_current_type(current_type)
    constant = float(constant)
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    _, _, zp = relative_coordinates(x, y, z, x_poles, y_poles, z_poles)

    rho = horizontal_distance(x, y, z, x_poles, y_poles, z_poles)

    coeff = constant / rho

    # G matrix scale factors
    if current_type == "divergence_free":
        rhohat = radial_unit_vectors(x, y, z, x_poles, y_poles, z_poles)

        Grhoprime = -coeff * (1 - np.abs(zp) / np.sqrt(rho**2 + zp**2)) * np.sign(zp)

        Gx = Grhoprime * rhohat[:, :, 0]
        Gy = Grhoprime * rhohat[:, :, 1]

        Gz = -coeff * rho / np.sqrt(rho**2 + zp**2)

    elif current_type == "curl_free":
        phihat = azimuth_unit_vectors(x, y, z, x_poles, y_poles, z_poles)

        heaviside = np.where((zp > 0.0) | np.isclose(zp, 0.0), 1.0, 0.0)

        Gphiprime = -2.0 * coeff * heaviside

        Gx = Gphiprime * phihat[:, :, 0]
        Gy = Gphiprime * phihat[:, :, 1]
        Gz = np.zeros_like(phihat[:, :, 2])

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
