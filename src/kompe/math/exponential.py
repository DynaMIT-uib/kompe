"""Exact propagation of constant-coefficient affine linear equations."""

import numpy as np

from kompe.math.backend import get_array_module, get_backend
from kompe.math.linear_map import as_linear_map, diagonal_linear_map


def affine_exponential(A, b, duration):
    """Return ``P, q`` such that ``x(t+h) = P(x(t)) + q`` for ``x' = A x + b``.

    ``A`` is a square linear map whose input and output scientific shapes
    agree. ``b`` has that same shape; ``duration`` is the non-negative
    interval h. The returned ``P`` is a LinearMap and ``q`` an array.

    Diagonal generators stay diagonal, including exactly zero rates.
    Otherwise exponentiate the augmented matrix ``[[A, b], [0, 0]]``:
    only one extra row and column are needed, not a full forcing-response
    matrix. This supports singular and non-normal A without an inverse
    or an equilibrium. A matrix-free A is materialized for this dense
    exponential. Construction uses SciPy on NumPy and JAX's expm on JAX;
    subsequent application of the returned map is backend-portable.
    """
    operator = as_linear_map(A)
    if operator.input_shape != operator.output_shape:
        raise ValueError("A must have the same input_shape and output_shape.")
    if isinstance(duration, (bool, np.bool_)):
        raise TypeError("duration must be a finite non-negative scalar.")
    duration = float(duration)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be a finite non-negative scalar.")
    xp = get_array_module(*operator.backend_operands, b)
    forcing = xp.asarray(b)
    if forcing.shape != operator.input_shape:
        raise ValueError(f"b must have shape {operator.input_shape}; got {forcing.shape}.")

    if operator.is_diagonal:
        z = duration * xp.asarray(operator.diagonal())
        # The removable singularity has phi_1(0) = 1. Do not divide by
        # zero even in an unselected JAX branch.
        denominator = xp.where(z == 0, 1, z)
        phi1 = xp.where(z == 0, 1, xp.expm1(z) / denominator)
        propagator = diagonal_linear_map(
            xp.exp(z), input_shape=operator.input_shape, output_shape=operator.output_shape
        )
        return propagator, (duration * phi1 * forcing.reshape(-1)).reshape(forcing.shape)

    matrix = operator.to_matrix(backend=get_backend(*operator.backend_operands, forcing))
    dtype = xp.result_type(matrix, forcing, 0.0)
    n = operator.shape[0]
    # A change of units for b must not inflate expm's scaling count.
    # Scale the auxiliary coordinate, then undo that exact similarity
    # transform on the forcing increment. No physical rate is shifted.
    forcing_scale = xp.maximum(xp.max(xp.abs(forcing), initial=0), 1)
    augmented = xp.concatenate(
        [
            xp.concatenate(
                [xp.asarray(matrix, dtype=dtype), forcing.reshape(n, 1) / forcing_scale], axis=1
            ),
            xp.zeros((1, n + 1), dtype=dtype),
        ],
        axis=0,
    )
    if xp is np:
        from scipy.linalg import expm
    else:
        from jax.scipy.linalg import expm

    exponential = expm(duration * augmented)
    propagator = as_linear_map(
        exponential[:n, :n],
        input_shape=operator.input_shape,
        output_shape=operator.output_shape,
    )
    return propagator, forcing_scale * exponential[:n, n].reshape(forcing.shape)
