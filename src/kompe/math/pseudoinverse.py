"""Backend-aware pseudoinverses preserving coefficient and data axes."""

from __future__ import annotations

import math

from kompe.math.backend import get_array_module, synchronize_linalg_result


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
    return synchronize_linalg_result(A_pinv).reshape(last_dims + first_dims)


def weighted_tensor_pinv(A, sqrt_weights=None, n_leading_flattened=2, rtol=1e-15):
    """Weighted Moore-Penrose pseudoinverse of a tensor."""
    if sqrt_weights is None:
        return tensor_pinv(A, n_leading_flattened=n_leading_flattened, rtol=rtol)

    xp = get_array_module(A, sqrt_weights)
    A_arr = xp.asarray(A)
    first_dims = A_arr.shape[:n_leading_flattened]
    last_dims = A_arr.shape[n_leading_flattened:]
    weights = xp.asarray(sqrt_weights).reshape(first_dims)
    weighted_A = weights.reshape(first_dims + (1,) * len(last_dims)) * A_arr
    return tensor_pinv(weighted_A, n_leading_flattened=n_leading_flattened, rtol=rtol) * weights
