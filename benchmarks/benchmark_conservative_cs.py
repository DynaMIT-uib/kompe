"""Investigate a weighted-adjoint global-CS discretization, not a production option.

With cell areas M and the current collocated gradient G, define
D = -M^-1 G* diag(M, M) and L = D G. This gives exact discrete integration
by parts, nonpositive energy, and conservation. It does NOT establish
pointwise consistency of D, freedom from spurious modes, or curl identities.
The benchmark deliberately checks accuracy independently of conservation.

Run from Kompe with PYTHONPATH=src. SciPy setup is CPU work; apply the fixed
operators with --backend numpy or jax. Timings are diagnostic, not CI limits.
"""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.special import eval_legendre, lpmv

from kompe import GlobalCSMesh
from kompe.math import as_linear_map, block_until_ready, get_array_module, set_backend


def weighted_adjoint_operators(mesh):
    """Return conservative D and D G using the existing collocated G."""
    gradient = sp.vstack(mesh.operators.surface_gradient_matrices(), format="csr")
    areas = mesh.cell_areas.reshape(-1)
    divergence = -sp.diags(1 / areas) @ gradient.T @ sp.diags(np.tile(areas, 2))
    return gradient, divergence.tocsr(), (divergence @ gradient).tocsr()


def main():
    """Compare accuracy, conservation, and cost without changing production."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if args.repeat < 1 or any(n < 4 or n % 2 for n in args.resolutions):
        parser.error("Use positive repeat and even resolutions >= 4.")
    if args.backend == "jax":
        import jax

        if not jax.config.x64_enabled:
            parser.error("Set JAX_ENABLE_X64=1 for double-precision comparisons.")
    set_backend(args.backend)
    xp = get_array_module()
    for resolution in args.resolutions:
        mesh = GlobalCSMesh(resolution)
        areas = mesh.cell_areas.reshape(-1)
        theta, phi = np.deg2rad(mesh.theta), np.deg2rad(mesh.phi)
        start = perf_counter()
        mesh.operators.surface_gradient_matrices()
        gradient_setup = perf_counter() - start
        start = perf_counter()
        gradient, divergence, conservative = weighted_adjoint_operators(mesh)
        conservative_setup = perf_counter() - start
        start = perf_counter()
        collocated = mesh.operators._unit_surface_laplacian_matrix
        collocated_setup = perf_counter() - start
        # Axisymmetric and non-axisymmetric harmonics probe face seams/poles.
        fields = np.column_stack(
            [
                np.cos(theta),
                eval_legendre(4, np.cos(theta)),
                lpmv(2, 3, np.cos(theta)) * np.cos(2 * phi),
            ]
        )
        exact = fields * np.asarray([-2.0, -20.0, -12.0])
        field_scale = np.sqrt(np.sum(areas[:, None] * exact**2, axis=0) / areas.sum())
        random = np.random.default_rng(72)
        scalar, vector = random.normal(size=mesh.size), random.normal(size=2 * mesh.size)
        lhs = scalar @ (areas * (divergence @ vector))
        rhs = -(gradient @ scalar) @ (np.tile(areas, 2) * vector)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(areas @ conservative, 0, atol=1e-11)
        assert scalar @ (areas * (conservative @ scalar)) <= 0

        # A small dense spectrum diagnoses extra/null or weak grid modes.
        smallest = None
        if mesh.size <= 500:
            metric = sp.diags(np.sqrt(areas))
            positive = -metric @ conservative @ sp.diags(1 / np.sqrt(areas))
            smallest = np.linalg.eigvalsh(positive.toarray())[:9].tolist()
        for name, matrix, setup in [
            ("collocated", collocated, collocated_setup),
            ("weighted_adjoint", conservative, conservative_setup),
        ]:
            operator = as_linear_map(matrix)
            values = xp.asarray(fields)
            actual = np.asarray(block_until_ready(operator(values)))
            durations = []
            for _ in range(args.repeat):
                start = perf_counter()
                block_until_ready(operator(values))
                durations.append(perf_counter() - start)
            relative_rms = (
                np.sqrt(np.sum(areas[:, None] * (actual - exact) ** 2, axis=0) / areas.sum())
                / field_scale
            )
            left_null = np.max(np.abs(areas @ matrix))
            constant = np.max(np.abs(matrix @ np.ones(mesh.size)))
            print(
                json.dumps(
                    {
                        "backend": args.backend,
                        "resolution": resolution,
                        "method": name,
                        "relative_rms_l1_l4_l3m2": relative_rms.tolist(),
                        "integrated_laplacian": (areas @ actual).tolist(),
                        "constant_residual": float(constant),
                        "area_left_null_residual": float(left_null),
                        "nonzeros": matrix.nnz,
                        "additional_assembly_s": setup,
                        "common_gradient_assembly_s": gradient_setup,
                        "apply_three_fields_s": median(durations),
                        "smallest_eigenvalues": smallest if name == "weighted_adjoint" else None,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
