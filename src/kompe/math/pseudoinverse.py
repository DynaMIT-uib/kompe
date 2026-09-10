"""Backend-aware pseudoinverses preserving coefficient and data axes."""

from __future__ import annotations

import math

from kompe.math.backend import get_array_module, synchronize_linalg_result


def tensor_pinv(A, output_ndim=1, rtol=1e-15, hermitian=False):
    """Invert a linear map whose array retains input and output axes.

    ``A.shape`` is ``output_shape + input_shape``; ``output_ndim``
    specifies where these shapes split. The inverse has shape
    ``input_shape + output_shape``. The default treats a matrix just
    like ``numpy.linalg.pinv``. For vector samples shaped
    ``(2, n_points, n_coefficients)``, pass ``output_ndim=2``.

    Empty input or output shapes represent scalar spaces. This is
    not a batched matrix inverse: all input axes and all output axes
    each belong to a single linear map.
    """
    xp = get_array_module(A)
    A_arr = xp.asarray(A)

    if not 0 <= output_ndim <= A_arr.ndim:
        raise ValueError("output_ndim must lie between zero and A.ndim.")
    output_shape = A_arr.shape[:output_ndim]
    input_shape = A_arr.shape[output_ndim:]
    matrix = A_arr.reshape(math.prod(output_shape), math.prod(input_shape))
    inverse = xp.linalg.pinv(matrix, rtol=rtol, hermitian=hermitian)
    return synchronize_linalg_result(inverse).reshape(input_shape + output_shape)


def weighted_tensor_pinv(A, sqrt_weights=None, output_ndim=1, rtol=1e-15):
    """Map samples to a weighted least-squares solution.

    Return ``pinv(W A) W``, where ``W`` is diagonal with the supplied
    square-root weights, one per output sample. Axis order and
    ``output_ndim`` follow :func:`tensor_pinv`. Weights may be supplied
    flat or with the output shape; omitted weights mean ordinary
    least squares.
    """
    if sqrt_weights is None:
        return tensor_pinv(A, output_ndim=output_ndim, rtol=rtol)

    xp = get_array_module(A, sqrt_weights)
    A_arr = xp.asarray(A)
    output_shape = A_arr.shape[:output_ndim]
    input_shape = A_arr.shape[output_ndim:]
    weights = xp.asarray(sqrt_weights).reshape(output_shape)
    weighted_A = weights.reshape(output_shape + (1,) * len(input_shape)) * A_arr
    return tensor_pinv(weighted_A, output_ndim=output_ndim, rtol=rtol) * weights
