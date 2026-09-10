"""Compare shared SH derivatives with pre-stacked gradient/rotated-gradient arrays.

Warm timings include single fields and batches, forward and adjoint actions.
JAX timings synchronize results and exclude compilation. Reported storage is
derivative storage, not peak process memory or an end-to-end simulation claim.
"""

import argparse
import json
import statistics
from time import perf_counter

import numpy as np

from kompe import SHBasis, SphericalGrid
from kompe.math import block_until_ready, get_array_module, set_backend


def main():
    """Time production synthesis against the previous stacked formulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--points", type=int, default=1728)
    parser.add_argument("--degree", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=15)
    args = parser.parse_args()
    set_backend(args.backend)
    xp = get_array_module()
    if args.backend == "jax":
        import jax

        if not jax.config.x64_enabled:
            parser.error("Set JAX_ENABLE_X64=1 for the float64 comparison")
    grid = SphericalGrid(
        theta=np.linspace(5, 175, args.points),
        phi=np.arange(args.points) * 137.508,
    )
    basis = SHBasis(args.degree, args.degree)
    operator = basis.helmholtz_synthesis_operator(grid)
    theta = basis.scalar_evaluation_array(grid, "theta")
    phi = basis.scalar_evaluation_array(grid, "phi")
    gradient = xp.concatenate([theta, phi])
    rotated = xp.concatenate([-phi, theta])
    n, m = basis.coefficient_count, grid.size

    def stacked_forward(c):
        return -gradient @ c[:n] + rotated @ c[n:]

    def stacked_adjoint(f):
        return xp.concatenate([-gradient.T @ f, rotated.T @ f])

    print(
        json.dumps(
            {
                **vars(args),
                "coefficients": n,
                "shared_derivative_bytes": theta.nbytes + phi.nbytes,
                "previous_derivative_bytes": theta.nbytes
                + phi.nbytes
                + gradient.nbytes
                + rotated.nbytes,
            }
        )
    )
    rng = np.random.default_rng(1028)
    for batch_size in (1, 16):
        for direction, production, reference, rows in (
            ("forward", operator.matmat, stacked_forward, 2 * n),
            ("adjoint", operator.rmatmat, stacked_adjoint, 2 * m),
        ):
            values = xp.asarray(rng.normal(size=(rows, batch_size)))
            expected = reference(values)
            for name, operation in (("stacked", reference), ("shared", production)):
                if args.backend == "jax":
                    operation = jax.jit(operation)
                result = block_until_ready(operation(values))
                np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
                times = []
                for _ in range(args.repeat):
                    start = perf_counter()
                    block_until_ready(operation(values))
                    times.append(perf_counter() - start)
                print(
                    json.dumps(
                        {
                            "batch_size": batch_size,
                            "direction": direction,
                            "formulation": name,
                            "median_s": statistics.median(times),
                        }
                    )
                )


if __name__ == "__main__":
    main()
