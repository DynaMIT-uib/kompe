"""Fused linear algebra for batches of cubed-sphere geometry matrices.

Metric tensors and coordinate Jacobians are 3-by-3 at every grid point.
Their explicit determinant and inverse avoid the overhead of a general
matrix factorization and fuse efficiently under JAX.
"""

from kompe.math.backend import get_array_module


def determinant_3x3(matrices):
    """Return the determinant of every 3-by-3 matrix in a batch."""
    xp = get_array_module(matrices)
    matrices = xp.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("Input array must have shape (N, 3, 3).")

    return (
        matrices[:, 0, 0] * matrices[:, 1, 1] * matrices[:, 2, 2]
        - matrices[:, 0, 0] * matrices[:, 1, 2] * matrices[:, 2, 1]
        - matrices[:, 0, 1] * matrices[:, 1, 0] * matrices[:, 2, 2]
        + matrices[:, 0, 1] * matrices[:, 1, 2] * matrices[:, 2, 0]
        + matrices[:, 0, 2] * matrices[:, 1, 0] * matrices[:, 2, 1]
        - matrices[:, 0, 2] * matrices[:, 1, 1] * matrices[:, 2, 0]
    )


def inverse_3x3(matrices):
    """Return the inverse of every invertible 3-by-3 matrix in a batch."""
    xp = get_array_module(matrices)
    matrices = xp.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("Input array must have shape (N, 3, 3).")
    determinant = determinant_3x3(matrices)

    row_0 = xp.stack(
        (
            matrices[:, 1, 1] * matrices[:, 2, 2] - matrices[:, 1, 2] * matrices[:, 2, 1],
            matrices[:, 0, 2] * matrices[:, 2, 1] - matrices[:, 0, 1] * matrices[:, 2, 2],
            matrices[:, 0, 1] * matrices[:, 1, 2] - matrices[:, 0, 2] * matrices[:, 1, 1],
        ),
        axis=1,
    )
    row_1 = xp.stack(
        (
            matrices[:, 1, 2] * matrices[:, 2, 0] - matrices[:, 1, 0] * matrices[:, 2, 2],
            matrices[:, 0, 0] * matrices[:, 2, 2] - matrices[:, 0, 2] * matrices[:, 2, 0],
            matrices[:, 0, 2] * matrices[:, 1, 0] - matrices[:, 0, 0] * matrices[:, 1, 2],
        ),
        axis=1,
    )
    row_2 = xp.stack(
        (
            matrices[:, 1, 0] * matrices[:, 2, 1] - matrices[:, 1, 1] * matrices[:, 2, 0],
            matrices[:, 0, 1] * matrices[:, 2, 0] - matrices[:, 0, 0] * matrices[:, 2, 1],
            matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0],
        ),
        axis=1,
    )
    adjugate = xp.stack((row_0, row_1, row_2), axis=1)
    return adjugate / determinant[:, None, None]
