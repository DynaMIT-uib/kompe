"""Measure real weighted, regularized SH fits through production solvers.

Report first use, repeated construction of a fresh fit after JAX warm-up,
and new right-hand sides reusing one fit. Basis evaluation is shared and
excluded. These are CPU/GPU-specific measurements, not CI thresholds.
"""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np

from kompe import SHBasis, SphericalGrid, SphericalTransform
from kompe.math import LeastSquaresSolver, block_until_ready, get_array_module, set_backend


def main():
    """Time unequal component weights, a smoothness penalty, and exact mean gauges."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--degree", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if min(args.degree, args.repeat) < 1:
        parser.error("degree and repeat must be positive")
    if args.backend == "jax":
        import jax

        if not jax.config.x64_enabled:
            parser.error("Set JAX_ENABLE_X64=1 for a float64 comparison")
    set_backend(args.backend)
    xp = get_array_module()
    grid = SphericalGrid(
        theta=np.linspace(3.0, 177.0, 24)[:, None], phi=np.arange(72)[None, :] * 5.0
    )
    basis = SHBasis(args.degree, args.degree, mean_free=False)
    rng = np.random.default_rng(322)
    weights = xp.asarray(rng.uniform(0.5, 2.0, (2, grid.size)))
    values = xp.asarray(rng.normal(size=(2, grid.size)))
    common = SphericalTransform(basis, grid, sqrt_weights=weights, reg_lambda=0.001)
    block_until_ready(common.gradient_theta_array)
    block_until_ready(common.gradient_phi_array)

    # Independent augmented least squares, removing constant SH modes
    # directly instead of using the production gauge or normal builder.
    synthesis = np.asarray(common.helmholtz_synthesis_array).reshape(2 * grid.size, -1)
    weighted = np.asarray(weights).reshape(-1, 1) * synthesis
    penalty = np.asarray(common.helmholtz_regularization_operator.diagonal())
    data_diag = np.sum(weighted**2, axis=0)
    penalty_diag = penalty**2
    scale = np.sqrt(
        0.001 * np.median(data_diag[data_diag > 0]) / np.median(penalty_diag[penalty_diag > 0])
    )
    keep = np.tile(basis.n != 0, 2)
    augmented = np.vstack([weighted, scale * np.diag(penalty)])[:, keep]
    rhs = np.r_[
        np.asarray(weights).reshape(-1) * np.asarray(values).reshape(-1), np.zeros(penalty.size)
    ]
    expected = np.zeros(penalty.size)
    expected[keep] = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    # The reference's explicit synthesis is not a production warm cache.
    common.helmholtz_synthesis_operator.clear_dense_cache()
    print(
        json.dumps(
            {
                **vars(args),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "points": grid.size,
                "scalar_coefficients": basis.coefficient_count,
                "scope": "unequal weights, relative regularization 0.001, exact mean gauges",
            }
        ),
        flush=True,
    )
    for method in ("normal_solve", "normal_pinv"):
        solver = LeastSquaresSolver(method, tolerance=1e-12)
        new_fit_times = []
        for index in range(args.repeat + 1):
            start = perf_counter()
            transform = SphericalTransform(basis, grid, sqrt_weights=weights, reg_lambda=0.001)
            result = block_until_ready(transform.analyze_helmholtz(values, solver=solver))
            elapsed = perf_counter() - start
            if index:
                new_fit_times.append(elapsed)
            else:
                first = elapsed
        np.testing.assert_allclose(np.asarray(result).reshape(-1), expected, rtol=1e-8, atol=1e-10)
        repeated_times = []
        for _ in range(args.repeat):
            start = perf_counter()
            repeated = block_until_ready(transform.analyze_helmholtz(2 * values, solver=solver))
            repeated_times.append(perf_counter() - start)
        np.testing.assert_allclose(repeated, 2 * result, rtol=1e-10, atol=1e-11)
        explicit_data = transform.helmholtz_least_squares_problem.data_operator.materialized_matrix
        print(
            json.dumps(
                {
                    "solver": method,
                    "first_fit_s": first,
                    "warm_new_fit_s": median(new_fit_times),
                    "cached_rhs_s": median(repeated_times),
                    "explicit_data_bytes": 0 if explicit_data is None else explicit_data.nbytes,
                    "relative_error": float(
                        np.linalg.norm(np.asarray(result).reshape(-1) - expected)
                        / np.linalg.norm(expected)
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
