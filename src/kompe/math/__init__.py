"""Backend-neutral mathematical helpers shared by spherical consumers."""

from kompe.math.backend import (
    JAX_AVAILABLE,
    backend_context,
    block_until_ready,
    get_array_module,
    get_backend,
    jax_enabled,
    readonly_numpy_array,
    set_backend,
    synchronize_linalg_result,
    to_numpy,
)
from kompe.math.einsum import einsum_linear_map, einsum_linear_map_from_matvec
from kompe.math.fingerprints import array_fingerprint, content_fingerprint
from kompe.math.finite_differences import finite_difference_weights
from kompe.math.least_squares_problem import LeastSquaresProblem
from kompe.math.least_squares_solver import (
    LEAST_SQUARES_SOLVER_ENV,
    LeastSquaresSolver,
    cholesky_least_squares_map,
    dense_full_rank_least_squares_factor,
    dense_full_rank_least_squares_map,
    get_default_least_squares_solver,
    sparse_constrained_least_squares_map,
)
from kompe.math.linear_map import (
    ArrayBackend,
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
    identity_linear_map,
    is_identity_linear_map,
    pointwise_matrix_linear_map,
    take_linear_map,
    vstack_linear_maps,
)
from kompe.math.pseudoinverse import tensor_pinv, weighted_tensor_pinv
from kompe.math.small_matrices import determinant_3x3, inverse_3x3

__all__ = [
    "JAX_AVAILABLE",
    "LEAST_SQUARES_SOLVER_ENV",
    "ArrayBackend",
    "LeastSquaresProblem",
    "LeastSquaresSolver",
    "LinearMap",
    "array_fingerprint",
    "as_linear_map",
    "backend_context",
    "block_until_ready",
    "cholesky_least_squares_map",
    "content_fingerprint",
    "dense_full_rank_least_squares_factor",
    "dense_full_rank_least_squares_map",
    "determinant_3x3",
    "diagonal_linear_map",
    "einsum_linear_map",
    "einsum_linear_map_from_matvec",
    "finite_difference_weights",
    "get_array_module",
    "get_backend",
    "get_default_least_squares_solver",
    "identity_linear_map",
    "inverse_3x3",
    "is_identity_linear_map",
    "jax_enabled",
    "pointwise_matrix_linear_map",
    "readonly_numpy_array",
    "set_backend",
    "sparse_constrained_least_squares_map",
    "synchronize_linalg_result",
    "take_linear_map",
    "tensor_pinv",
    "to_numpy",
    "vstack_linear_maps",
    "weighted_tensor_pinv",
]
