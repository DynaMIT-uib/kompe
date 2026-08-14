"""Backend-neutral mathematical helpers shared by spherical consumers."""

from kompe.math.backend import (
    JAX_AVAILABLE,
    asarray,
    backend_context,
    block_after_jax_linalg,
    block_until_ready,
    get_array_module,
    jit,
    set_backend,
    to_jax,
    to_numpy,
    use_jax,
    vmap,
)
from kompe.math.einsum import einsum_linear_map, einsum_linear_map_from_matvec
from kompe.math.fingerprints import array_fingerprint, content_fingerprint
from kompe.math.least_squares_problem import LeastSquaresProblem
from kompe.math.least_squares_solver import (
    LEAST_SQUARES_SOLVER_ENV,
    LeastSquaresSolver,
    dense_full_rank_least_squares_factor,
    dense_full_rank_least_squares_map,
    factorized_least_squares_map,
    get_default_least_squares_solver,
    sparse_constrained_least_squares_map,
)
from kompe.math.linear_map import (
    LinearMap,
    MatrixBackend,
    as_linear_map,
    diagonal_linear_map,
    identity_linear_map,
    is_noop_linear_map,
    pointwise_matrix_linear_map,
    take_linear_map,
    vstack_linear_maps,
)
from kompe.math.tensor_operations import tensor_pinv, weighted_tensor_pinv

__all__ = [
    "JAX_AVAILABLE",
    "LEAST_SQUARES_SOLVER_ENV",
    "LeastSquaresProblem",
    "LeastSquaresSolver",
    "LinearMap",
    "MatrixBackend",
    "array_fingerprint",
    "as_linear_map",
    "asarray",
    "backend_context",
    "block_after_jax_linalg",
    "block_until_ready",
    "content_fingerprint",
    "dense_full_rank_least_squares_factor",
    "dense_full_rank_least_squares_map",
    "diagonal_linear_map",
    "einsum_linear_map",
    "einsum_linear_map_from_matvec",
    "factorized_least_squares_map",
    "get_array_module",
    "get_default_least_squares_solver",
    "identity_linear_map",
    "is_noop_linear_map",
    "jit",
    "pointwise_matrix_linear_map",
    "set_backend",
    "sparse_constrained_least_squares_map",
    "take_linear_map",
    "tensor_pinv",
    "to_jax",
    "to_numpy",
    "use_jax",
    "vmap",
    "vstack_linear_maps",
    "weighted_tensor_pinv",
]
