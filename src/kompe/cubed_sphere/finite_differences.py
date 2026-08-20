"""Finite-difference weights used by regional cubed-sphere operators."""

from fractions import Fraction
from math import factorial

import numpy as np


def _least_common_multiple(values):
    """Return the least common multiple of an integer sequence."""
    return np.lcm.reduce(np.asarray(values, dtype=int))


def finite_difference_weights(stencil_points, order=1, h=1, fraction=False):
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
    fraction: bool, optional
        Set to True to return coefficients as integer numerators
        and a common denominator. Be careful with this for large stencils.

    Returns
    -------
    coefficients: array
        array of finite-difference coefficients. Unless fraction is set
        to True - in which case a tuple will be returned with
        an array of numerators and an integer denominator. If
        fraction is True, h is ignored - and you should multiply the
        denominator by h**order to get the coefficients

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

    if fraction:
        fractions = [Fraction(weight).limit_denominator() for weight in weights]
        common_denominator = _least_common_multiple(
            [value.denominator for value in fractions]
        )
        numerators = [
            value.numerator * common_denominator // value.denominator
            for value in fractions
        ]
        return numerators, common_denominator
    return weights / h**order


__all__ = ["finite_difference_weights"]
