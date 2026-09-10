"""Compare explicit and blockwise weighted Helmholtz normal matrices.

Synthetic real derivative blocks isolate the linear algebra. The structured
case calls the production builder, not a separate prototype implementation.
JAX compilation is excluded from warm timings; array results are synchronized.
"""

import argparse
import json
import platform
import statistics
from functools import partial
from time import perf_counter

import numpy as np

from kompe.basis import _helmholtz_normal_matrix
from kompe.math import get_array_module, set_backend


def explicit_normal(theta, phi, sqrt_weights, xp):
    """Form H* W**2 H through the explicit Helmholtz synthesis matrix."""
    synthesis = xp.block([[-theta, -phi], [-phi, theta]])
    weighted = sqrt_weights.reshape(-1, 1) * synthesis
    return weighted.T @ weighted


def main():
    """Compare structured production assembly with an explicit reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--points", type=int, default=1728)
    parser.add_argument("--coefficients", type=int, default=440)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    if min(args.points, args.coefficients, args.repeat) < 1:
        parser.error("points, coefficients, and repeat must be positive")
    if args.backend == "jax":
        import jax

        if not jax.config.x64_enabled:
            parser.error("Set JAX_ENABLE_X64=1 for the float64 comparison")
    set_backend(args.backend)
    xp = get_array_module()
    rng = np.random.default_rng(204)
    arrays = (
        xp.asarray(rng.normal(size=(args.points, args.coefficients))),
        xp.asarray(rng.normal(size=(args.points, args.coefficients))),
        xp.asarray(rng.uniform(0.5, 2.0, size=(2, args.points))),
    )
    reference = explicit_normal(*(np.asarray(value) for value in arrays), np)
    print(
        json.dumps(
            {
                **vars(args),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "rectangular_synthesis_bytes": 4 * arrays[0].nbytes,
                "note": "Synthetic float64 blocks; not end-to-end or GPU speedup measurements.",
            }
        )
    )
    for name, operation in (
        ("explicit", partial(explicit_normal, xp=xp)),
        ("structured", _helmholtz_normal_matrix),
    ):
        temporary_bytes = None
        if args.backend == "jax":
            operation = jax.jit(operation).lower(*arrays).compile()
            temporary_bytes = operation.memory_analysis().temp_size_in_bytes
        result = operation(*arrays)
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=2e-11)
        times = []
        for _ in range(args.repeat):
            start = perf_counter()
            result = operation(*arrays)
            if args.backend == "jax":
                result.block_until_ready()
            times.append(perf_counter() - start)
        print(
            json.dumps(
                {
                    "formulation": name,
                    "median_s": statistics.median(times),
                    "jax_compiler_temporary_bytes": temporary_bytes,
                    "max_abs_error": float(np.max(np.abs(np.asarray(result) - reference))),
                }
            )
        )


if __name__ == "__main__":
    main()
