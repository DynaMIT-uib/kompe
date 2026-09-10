"""Time repeated compiled LSMR or sparse regularized CS fits.

Select NumPy/JAX with KOMPE_USE_JAX and request JAX_ENABLE_X64=1 for
double-precision comparisons. Run revisions in separate Python processes.
"""

import argparse
import json
import time
from unittest.mock import patch

import numpy as np

from kompe import GlobalCSBasis, SphericalTransform
from kompe.math import LeastSquaresProblem, LeastSquaresSolver, LinearMap, get_array_module
from kompe.math.backend import block_until_ready

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case", choices=["lsmr", "cs"])
parser.add_argument("--resolution", type=int, default=16)
args = parser.parse_args()
xp = get_array_module()
rng = np.random.default_rng(8)

if args.case == "lsmr":
    A = xp.asarray(rng.normal(size=(100, 60)))
    rhs = xp.asarray(rng.normal(size=(100, 8)))
    problem = LeastSquaresProblem(A)
    solver = LeastSquaresSolver("lsmr", tolerance=1e-10)

    def solve():
        """Solve the same dense-operator fit for eight independent fields."""
        return solver.solve(problem, rhs)

else:
    basis = GlobalCSBasis(args.resolution)
    transform = SphericalTransform(basis, basis.native_grid, area_weighted=True, reg_lambda=0.01)
    rhs = xp.asarray(rng.normal(size=(2, basis.coefficient_count, 4)))

    def solve():
        """Fit regularized native CS fields with sparse normal equations."""
        return transform.analyze_helmholtz(rhs, solver="normal_solve")


materializations = []
original = LinearMap.to_matrix


def record(operator, **kwargs):
    """Record explicit matrix requests without changing their implementation."""
    materializations.append(operator.shape)
    return original(operator, **kwargs)


with patch.object(LinearMap, "to_matrix", record):
    start = time.perf_counter()
    result = block_until_ready(solve())
    cold = time.perf_counter() - start
    warm = []
    for _ in range(5):
        start = time.perf_counter()
        block_until_ready(solve())
        warm.append(time.perf_counter() - start)

print(
    json.dumps(
        {
            "case": args.case,
            "backend": xp.__name__,
            "resolution": args.resolution,
            "cold_s": cold,
            "warm_median_s": float(np.median(warm)),
            "dense_shapes": materializations,
            "solution_norm": float(xp.linalg.norm(result)),
        },
        indent=2,
    )
)
