"""Finite-difference stencil weights for CPU operator construction."""

from math import factorial

import numpy as np


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


__all__ = ["finite_difference_weights"]
