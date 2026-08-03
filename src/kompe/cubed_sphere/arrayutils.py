"""Array utilities.

This module contains utility functions for performing array operations
such as computing determinants and inverses of 3D matrices, as well as
constraining array values within specified bounds.
"""

import numpy as np


def determinants_3x3(matrices):
    """Calculate determinants of stacked 3-by-3 matrices.

    Parameters
    ----------
    matrices : array
        Array with shape ``(N, 3, 3)``, corresponding to ``N`` 3D
        matrices.

    Returns
    -------
    det : array
        Array with determinants, shape ``(N)``.

    Raises
    ------
    ValueError
        If the input array is not 3D or if the last two axes are not
        3 x 3.
    """
    matrices = np.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("Input array must have shape (N, 3, 3).")

    det = (
        matrices[:, 0, 0] * matrices[:, 1, 1] * matrices[:, 2, 2]
        - matrices[:, 0, 0] * matrices[:, 1, 2] * matrices[:, 2, 1]
        - matrices[:, 0, 1] * matrices[:, 1, 0] * matrices[:, 2, 2]
        + matrices[:, 0, 1] * matrices[:, 1, 2] * matrices[:, 2, 0]
        + matrices[:, 0, 2] * matrices[:, 1, 0] * matrices[:, 2, 1]
        - matrices[:, 0, 2] * matrices[:, 1, 1] * matrices[:, 2, 0]
    )

    return det


def invert_3x3_matrices(matrices):
    """Calculate inverses of stacked 3-by-3 matrices.

    Parameters
    ----------
    matrices : array
        Array with shape ``(N, 3, 3)``, corresponding to ``N`` 3D
        invertible matrices.

    Returns
    -------
    Minv : array
        Array with inverse matrices, shape ``(N, 3, 3)``.

    Raises
    ------
    ValueError
        If the input array is not 3D, if the last two axes are not
        3 x 3, or if any of the matrices are not invertible.
    """
    matrices = np.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("Input array must have shape (N, 3, 3).")
    det = determinants_3x3(matrices)

    # Determinants carry physical scaling. For example, Cartesian-to-CS
    # Jacobians have determinant proportional to ``r**-2`` and are perfectly
    # invertible at Earth radius even though an absolute ``isclose`` check
    # classifies them as zero.
    singular = (det == 0) | ~np.isfinite(det)
    if np.any(singular):
        raise ValueError(f"The following matrices are not invertible: {np.where(singular)[0]}.")

    inverses = np.empty(matrices.shape)
    inverses[:, 0, 0] = (
        matrices[:, 1, 1] * matrices[:, 2, 2] - matrices[:, 1, 2] * matrices[:, 2, 1]
    )
    inverses[:, 0, 1] = (
        -matrices[:, 0, 1] * matrices[:, 2, 2] + matrices[:, 0, 2] * matrices[:, 2, 1]
    )
    inverses[:, 0, 2] = (
        matrices[:, 0, 1] * matrices[:, 1, 2] - matrices[:, 0, 2] * matrices[:, 1, 1]
    )
    inverses[:, 1, 0] = (
        -matrices[:, 1, 0] * matrices[:, 2, 2] + matrices[:, 1, 2] * matrices[:, 2, 0]
    )
    inverses[:, 1, 1] = (
        matrices[:, 0, 0] * matrices[:, 2, 2] - matrices[:, 0, 2] * matrices[:, 2, 0]
    )
    inverses[:, 1, 2] = (
        -matrices[:, 0, 0] * matrices[:, 1, 2] + matrices[:, 0, 2] * matrices[:, 1, 0]
    )
    inverses[:, 2, 0] = (
        matrices[:, 1, 0] * matrices[:, 2, 1] - matrices[:, 1, 1] * matrices[:, 2, 0]
    )
    inverses[:, 2, 1] = (
        -matrices[:, 0, 0] * matrices[:, 2, 1] + matrices[:, 0, 1] * matrices[:, 2, 0]
    )
    inverses[:, 2, 2] = (
        matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
    )

    return inverses / det.reshape((matrices.shape[0], 1, 1))
