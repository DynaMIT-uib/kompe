"""Tensor operations module.

Backend-aware tensor contraction and pseudoinverse helpers.
"""

from __future__ import annotations

import math

from kompe.math.backend import block_after_jax_linalg, get_array_module


def tensor_pinv(A, n_leading_flattened=2, rtol=1e-15, hermitian=False):
    """Moore-Penrose pseudoinverse of a tensor."""
    xp = get_array_module(A)
    A_arr = xp.asarray(A)

    first_dims = A_arr.shape[:n_leading_flattened]
    last_dims = A_arr.shape[n_leading_flattened:]

    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    A_flat = A_arr.reshape((flat_first, flat_last))
    A_pinv = xp.linalg.pinv(A_flat, rtol=rtol, hermitian=hermitian)
    return block_after_jax_linalg(A_pinv).reshape(last_dims + first_dims)


def weighted_tensor_pinv(A, sqrt_weights=None, n_leading_flattened=2, rtol=1e-15):
    """Weighted Moore-Penrose pseudoinverse of a tensor."""
    if sqrt_weights is None:
        return tensor_pinv(A, n_leading_flattened=n_leading_flattened, rtol=rtol)

    xp = get_array_module(A, sqrt_weights)
    A_arr = xp.asarray(A)
    weights = xp.asarray(sqrt_weights)

    first_dims = A_arr.shape[:n_leading_flattened]
    last_dims = A_arr.shape[n_leading_flattened:]
    flat_first = math.prod(first_dims)
    flat_last = math.prod(last_dims)

    weights_flat = weights.reshape(flat_first)
    A_flat = A_arr.reshape((flat_first, flat_last))
    weighted_A = weights_flat.reshape((-1, 1)) * A_flat
    weighted_pinv = xp.linalg.pinv(weighted_A, rtol=rtol)
    weighted_pinv = block_after_jax_linalg(weighted_pinv)
    return (weighted_pinv * weights_flat.reshape((1, -1))).reshape(last_dims + first_dims)
