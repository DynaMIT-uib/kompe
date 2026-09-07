"""Fused determinant and inverse formulas for batches of small matrices.

The explicit 3-by-3 formulas avoid a general matrix factorization and
fuse efficiently under JAX. Cubed-sphere metrics and Jacobians are one
application; the routines do not depend on a coordinate system.
"""

from kompe.math.backend import get_array_module


def determinant_3x3(matrices):
    """Return determinants for an array of shape ``(..., 3, 3)``."""
    xp = get_array_module(matrices)
    matrices = xp.asarray(matrices)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("Input array must have shape (..., 3, 3).")

    return (
        matrices[..., 0, 0] * matrices[..., 1, 1] * matrices[..., 2, 2]
        - matrices[..., 0, 0] * matrices[..., 1, 2] * matrices[..., 2, 1]
        - matrices[..., 0, 1] * matrices[..., 1, 0] * matrices[..., 2, 2]
        + matrices[..., 0, 1] * matrices[..., 1, 2] * matrices[..., 2, 0]
        + matrices[..., 0, 2] * matrices[..., 1, 0] * matrices[..., 2, 1]
        - matrices[..., 0, 2] * matrices[..., 1, 1] * matrices[..., 2, 0]
    )


def inverse_3x3(matrices):
    """Invert nonsingular matrices in an array of shape ``(..., 3, 3)``."""
    xp = get_array_module(matrices)
    matrices = xp.asarray(matrices)
    determinant = determinant_3x3(matrices)

    row_0 = xp.stack(
        (
            matrices[..., 1, 1] * matrices[..., 2, 2] - matrices[..., 1, 2] * matrices[..., 2, 1],
            matrices[..., 0, 2] * matrices[..., 2, 1] - matrices[..., 0, 1] * matrices[..., 2, 2],
            matrices[..., 0, 1] * matrices[..., 1, 2] - matrices[..., 0, 2] * matrices[..., 1, 1],
        ),
        axis=-1,
    )
    row_1 = xp.stack(
        (
            matrices[..., 1, 2] * matrices[..., 2, 0] - matrices[..., 1, 0] * matrices[..., 2, 2],
            matrices[..., 0, 0] * matrices[..., 2, 2] - matrices[..., 0, 2] * matrices[..., 2, 0],
            matrices[..., 0, 2] * matrices[..., 1, 0] - matrices[..., 0, 0] * matrices[..., 1, 2],
        ),
        axis=-1,
    )
    row_2 = xp.stack(
        (
            matrices[..., 1, 0] * matrices[..., 2, 1] - matrices[..., 1, 1] * matrices[..., 2, 0],
            matrices[..., 0, 1] * matrices[..., 2, 0] - matrices[..., 0, 0] * matrices[..., 2, 1],
            matrices[..., 0, 0] * matrices[..., 1, 1] - matrices[..., 0, 1] * matrices[..., 1, 0],
        ),
        axis=-1,
    )
    adjugate = xp.stack((row_0, row_1, row_2), axis=-2)
    return adjugate / determinant[..., None, None]
