"""Finite-difference stencils and derivatives of sampled arrays."""

from math import factorial

import numpy as np

from kompe.math.backend import _is_jax_tracer, get_array_module


def finite_difference_weights(stencil_points, order=1, h=1):
    """Calculate weights for a finite-difference derivative.

    Parameters
    ----------
    stencil_points: array_like
        Sample positions relative to the evaluation point, for example
        ``[-1, 0, 1]`` for a central difference.
    order: integer, optional
        order of the derivative. Default 1 (first order)
    h: scalar, optional
        Step size. Default 1

    Returns
    -------
    coefficients: array
        Array of finite-difference coefficients.

    Note
    ----
    Algorithm from the Finite Difference Coefficient Calculator
    (https://web.media.mit.edu/~crtaylor/calculator.html)
    """
    stencil_points = np.asarray(stencil_points).reshape(1, -1)
    powers = np.arange(stencil_points.size).reshape(-1, 1)
    derivative = np.zeros(stencil_points.size)
    derivative[order] = factorial(order)
    weights = np.linalg.solve(stencil_points**powers, derivative)

    return weights / h**order


def centered_derivative(coordinates, values, *, half_window_points=1):
    """Differentiate samples along their last axis on an irregular grid.

    At each interior sample, differentiate the quadratic through that
    sample and its two neighbours, `half_window_points` indices away.
    Coordinates must be finite and strictly increasing. Values may have
    any leading batch axes. Incomplete or missing three-point stencils
    produce NaN. Arrays remain on their NumPy or JAX backend.

    Coordinates carry the independent variable's units; returned values
    have units of values divided by coordinates. No time/calendar policy
    or resampling is implied.
    """
    xp = get_array_module(coordinates, values)
    x = xp.asarray(coordinates, dtype=float)
    values = xp.asarray(values)
    values = xp.asarray(values, dtype=xp.result_type(values.dtype, 0.0))
    if x.ndim != 1 or values.ndim == 0 or values.shape[-1] != x.size:
        raise ValueError("The last values axis must match one-dimensional coordinates.")
    if not _is_jax_tracer(x) and not bool(xp.all(xp.isfinite(x)) & xp.all(xp.diff(x) > 0)):
        raise ValueError("coordinates must be finite and strictly increasing.")
    h = int(half_window_points)
    if isinstance(half_window_points, (bool, np.bool_)) or h != half_window_points or h < 1:
        raise ValueError("half_window_points must be a positive integer.")
    if x.size <= 2 * h:
        return xp.full_like(values, xp.nan)

    dx_left = x[h:-h] - x[: -2 * h]
    dx_right = x[2 * h :] - x[h:-h]
    center = values[..., h:-h]
    left_slope = (center - values[..., : -2 * h]) / dx_left
    right_slope = (values[..., 2 * h :] - center) / dx_right
    interior = (dx_right * left_slope + dx_left * right_slope) / (dx_left + dx_right)
    edges = xp.full(values.shape[:-1] + (h,), xp.nan, dtype=values.dtype)
    return xp.concatenate((edges, interior, edges), axis=-1)


__all__ = ["centered_derivative", "finite_difference_weights"]
