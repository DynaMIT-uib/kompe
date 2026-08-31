"""Spherical-harmonic normalization helpers."""

import math

import numpy as np


def schmidt_quasi_normalization_factors(max_degree: int, max_order: int):
    """
    Return a matrix of Schmidt quasi-normalization factors.

    The factors are computed according to the geomagnetism convention
    (e.g., Langel, 1987).

    Parameters
    ----------
    max_degree : int
        Maximum degree.
    max_order : int
        Maximum order.

    Returns
    -------
    S_matrix : ndarray, shape (max_degree+1, max_order+1)
        Matrix of normalization factors where S_matrix[n, m] is the
        factor for the (n, m) pair.
    """
    S_matrix = np.zeros((max_degree + 1, max_order + 1))
    S_matrix[0, 0] = 1.0

    for n in range(1, max_degree + 1):
        # Recurrence for m=0
        S_matrix[n, 0] = S_matrix[n - 1, 0] * (2.0 * n - 1.0) / n

        # Recurrence for m > 0
        for m in range(1, min(n, max_order) + 1):
            factor_m_dep = 2.0 if m == 1 else 1.0
            factor = math.sqrt((n - m + 1.0) * factor_m_dep / (n + m))
            S_matrix[n, m] = S_matrix[n, m - 1] * factor

    return S_matrix
