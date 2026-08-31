"""Backend-neutral mathematical helpers shared by spherical consumers."""

from kompe.math.backend import (
    JAX_AVAILABLE,
    backend_context,
    block_until_ready,
    get_array_module,
    get_backend,
    jax_enabled,
    set_backend,
    synchronize_linalg_result,
    to_numpy,
)
from kompe.math.einsum import einsum_linear_map, einsum_linear_map_from_matvec
from kompe.math.fingerprints import array_fingerprint, content_fingerprint
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

__all__ = [
    "JAX_AVAILABLE",
    "LEAST_SQUARES_SOLVER_ENV",
    "LeastSquaresProblem",
    "LeastSquaresSolver",
    "LinearMap",
    "ArrayBackend",
    "array_fingerprint",
    "as_linear_map",
    "backend_context",
    "block_until_ready",
    "cholesky_least_squares_map",
    "content_fingerprint",
    "dense_full_rank_least_squares_factor",
    "dense_full_rank_least_squares_map",
    "diagonal_linear_map",
    "einsum_linear_map",
    "einsum_linear_map_from_matvec",
    "get_array_module",
    "get_backend",
    "get_default_least_squares_solver",
    "identity_linear_map",
    "is_identity_linear_map",
    "jax_enabled",
    "pointwise_matrix_linear_map",
    "set_backend",
    "sparse_constrained_least_squares_map",
    "synchronize_linalg_result",
    "take_linear_map",
    "tensor_pinv",
    "to_numpy",
    "vstack_linear_maps",
    "weighted_tensor_pinv",
]
