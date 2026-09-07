"""Small, dependency-free performance checks for Kompe's core paths."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from kompe import (
    GlobalCSBasis,
    RegionalCSMesh,
    RegionalCSProjection,
    SECSBasis,
    SHBasis,
    SphericalGrid,
)
from kompe.math import as_linear_map, block_until_ready, get_array_module, set_backend


def measure(label, operation, repeat):
    """Report the first call separately from repeated calls."""
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = operation()
        block_until_ready(result)
        samples.append(time.perf_counter() - start)
    print(f"{label:36s} first={samples[0]:.6f}s warm median={statistics.median(samples[1:]):.6f}s")


def main():
    """Run representative materialization benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cs-resolution", type=int, default=16)
    parser.add_argument("--sh-degree", type=int, default=20)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("--repeat must be at least 2 to measure first and warm calls.")
    set_backend(args.backend)
    xp = get_array_module()

    # A grid contraction reduced to coefficient space should be reused
    # when that map becomes part of a larger scientific calculation.
    rng = np.random.default_rng(12)
    left = as_linear_map(xp.asarray(rng.normal(size=(64, 2048))))
    right = as_linear_map(xp.asarray(rng.normal(size=(2048, 48))))
    contracted = left @ right
    measure("Coefficient contraction", contracted.to_matrix, args.repeat)
    composed = contracted @ as_linear_map(xp.asarray(rng.normal(size=(48, 40))))
    probe = xp.ones(40)
    measure("Materialized-map composition apply", lambda: composed @ probe, args.repeat)

    measure("GlobalCSBasis construction", lambda: GlobalCSBasis(args.cs_resolution), args.repeat)
    cs_basis = GlobalCSBasis(args.cs_resolution)
    measure("Global CS Laplacian operator", cs_basis.surface_laplacian_operator, args.repeat)

    target = SphericalGrid(
        lat=np.linspace(-80.0, 80.0, 1000),
        lon=np.linspace(-180.0, 180.0, 1000),
    )
    sh_basis = SHBasis(args.sh_degree, args.sh_degree)
    measure("SH scalar evaluation", lambda: sh_basis.scalar_evaluation_array(target), args.repeat)

    regional = RegionalCSMesh(
        RegionalCSProjection((20.0, 70.0), 0.0),
        1800.0,
        1400.0,
        shape=(80, 60),
        radius=6371.2,
    )
    measure(
        "Regional theta/phi gradient matrices",
        lambda: regional.operators.surface_gradient_matrices(sparse=True),
        args.repeat,
    )

    poles = SphericalGrid(lat=regional.lat.reshape(-1)[::12], lon=regional.lon.reshape(-1)[::12])
    secs = SECSBasis(poles=poles, current_type="divergence_free")
    measure(
        "SECS current array",
        lambda: secs.surface_current_array(target, singularity_limit=1.0),
        args.repeat,
    )
    chunked_secs = secs.surface_current_operator(target, singularity_limit=1.0, chunk_size=100)
    measure(
        "SECS chunked operator apply",
        lambda: chunked_secs @ np.ones(secs.index_length),
        args.repeat,
    )


if __name__ == "__main__":
    main()
