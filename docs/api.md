# Public API guide

Kompe's top-level namespace contains representation and geometry types. The
backend-neutral numerical layer is intentionally namespaced under
`kompe.math`; low-level SECS kernels are under `kompe.secs`.

## Representations and meshes

- `Grid`: immutable spherical sample points, optionally with area weights.
- `SHBasis`: real spherical-harmonic scalar and Helmholtz expansion.
- `SECSBasis`: curl-free or divergence-free elementary-current expansion;
  construction requires an explicit `current_type`. Its current operator
  accepts `chunk_size` for bounded-memory forward and adjoint evaluation.
- `GlobalCSBasis`: closed-sphere, cell-centred cubed-sphere expansion. Its
  resolution `N` is required.
- `GlobalCSMesh`: immutable six-face geometry used by `GlobalCSBasis`.
- `RegionalCSProjection`: rotated gnomonic coordinate chart.
- `RegionalCSGrid`: structured bounded mesh with an explicit radius.
- `RegionalCSGridSpec`: versioned serialization boundary for regional grids.
- `RegionalCSOperators`: available as `grid.operators`; owns interpolation,
  gradients, divergence, and metric-density calculations.

## Analysis and synthesis

`SphericalTransform(basis, grid)` binds a closed-surface basis to evaluation
points. `synthesize_scalar` and `synthesize_helmholtz` map coefficients to
samples. `analyze_scalar` and `analyze_helmholtz` fit samples already on the
bound grid. `analyze_scalar_samples` and `analyze_helmholtz_samples` accept
external sample grids, optionally remapping through `grid_remap_basis`, and
return batch-first coefficient rows.

The word *projection* in Kompe's type vocabulary refers to coordinate charts.
Mean-free coefficient projections retain that mathematical name because they
are explicit projections within one coefficient space.

## Numerical layer

`kompe.math.LinearMap` is the common structured-operator contract. Use
`as_linear_map` to wrap dense or sparse arrays and the named constructors for
diagonal, identity, pointwise, stacked, or indexed maps. Least-squares helpers
consume `LinearMap` without requiring dense materialization.

NumPy/SciPy is the reference backend. JAX is optional and lazy. Select it with
`kompe.math.set_backend("jax")`, a `backend_context("jax")`, or explicit JAX
arrays. Kompe never changes JAX's global 64-bit setting.

## Cache lifecycle

`GlobalCSBasis` and `SphericalTransform` expose `cache_info()` and
`clear_cache()`. Caches use exact representation/grid signatures; shared CS
remap matrices and per-basis target-grid caches are bounded. Long-running
applications can clear them at known lifecycle boundaries without changing
numerical results.
