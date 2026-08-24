"""Array utilities.

This module contains utility functions for performing array operations
such as computing determinants and inverses of 3D matrices, as well as
constraining array values within specified bounds.
"""

from kompe.math.backend import get_array_module


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
    xp = get_array_module(matrices)
    matrices = xp.asarray(matrices)
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
        If the input array is not 3D or if the last two axes are not
        3 x 3. The input matrices are assumed to be invertible.
    """
    xp = get_array_module(matrices)
    matrices = xp.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("Input array must have shape (N, 3, 3).")
    det = determinants_3x3(matrices)

    row_0 = xp.stack(
        (
            matrices[:, 1, 1] * matrices[:, 2, 2]
            - matrices[:, 1, 2] * matrices[:, 2, 1],
            matrices[:, 0, 2] * matrices[:, 2, 1]
            - matrices[:, 0, 1] * matrices[:, 2, 2],
            matrices[:, 0, 1] * matrices[:, 1, 2]
            - matrices[:, 0, 2] * matrices[:, 1, 1],
        ),
        axis=1,
    )
    row_1 = xp.stack(
        (
            matrices[:, 1, 2] * matrices[:, 2, 0]
            - matrices[:, 1, 0] * matrices[:, 2, 2],
            matrices[:, 0, 0] * matrices[:, 2, 2]
            - matrices[:, 0, 2] * matrices[:, 2, 0],
            matrices[:, 0, 2] * matrices[:, 1, 0]
            - matrices[:, 0, 0] * matrices[:, 1, 2],
        ),
        axis=1,
    )
    row_2 = xp.stack(
        (
            matrices[:, 1, 0] * matrices[:, 2, 1]
            - matrices[:, 1, 1] * matrices[:, 2, 0],
            matrices[:, 0, 1] * matrices[:, 2, 0]
            - matrices[:, 0, 0] * matrices[:, 2, 1],
            matrices[:, 0, 0] * matrices[:, 1, 1]
            - matrices[:, 0, 1] * matrices[:, 1, 0],
        ),
        axis=1,
    )
    adjugate = xp.stack((row_0, row_1, row_2), axis=1)
    return adjugate / det[:, None, None]
