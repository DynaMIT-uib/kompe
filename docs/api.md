# Public API guide

Kompe's top-level namespace contains representation and geometry types. The
backend-neutral numerical layer is intentionally namespaced under
`kompe.math`; low-level SECS kernels are under `kompe.secs`.

## Representations and meshes

- `SphericalGrid`: immutable spherical sample points in the caller's coordinate
  frame, optionally with area weights. Coordinates are stored flat while
  `shape` retains the broadcast input shape for reshaping evaluated arrays.
- `SHBasis`: real spherical-harmonic scalar and Helmholtz expansion.
- `SECSBasis`: curl-free or divergence-free elementary-current expansion;
  construction requires an explicit `current_type`, and every synthesis
  method follows that mode. Its current operator accepts `chunk_size` for
  bounded-memory forward and adjoint evaluation.
- `GlobalCSBasis`: closed-sphere, cell-centred cubed-sphere expansion. Its
  `cells_per_face` resolution is required. `interpolate_vector()` accepts and
  returns canonical `(theta, phi, radial)` components.
- `GlobalCSMesh`: immutable six-face geometry used by `GlobalCSBasis`.
- `GlobalCSProjection`: stateless six-face coordinate and component maps. Its
  physical vector transforms explicitly map ENU components to and from local
  `(xi, eta, radial)` components.
- `RegionalCSProjection`: rotated regional gnomonic coordinate chart.
- `RegionalCSMesh`: structured bounded mesh with an explicit radius.
- `RegionalCSMeshSpec`: versioned serialization boundary for regional meshes.

`RegionalCSMesh(..., shape=(n_eta, n_xi))` follows NumPy array order. When
specifying physical resolution, use the named `xi_cell_size=` and
`eta_cell_size=` keywords; xi is parallel to the projection orientation and eta
is perpendicular to it. `RegionalCSMesh.from_edges(...)` is the exact-geometry
constructor used by serialization and derived meshes.
- `RegionalCSOperators`: available as `mesh.operators`; owns interpolation,
  gradients, and divergence. Its public surface
  vectors use `(theta, phi)`; `coordinate_derivative_matrices()` explicitly
  exposes derivatives with respect to the local `(xi, eta)` chart.
- `SolidHarmonicOperators`: regular/irregular radial continuation for an
  `SHBasis`; these are not surface-basis operations.

## Analysis and synthesis

`SphericalTransform(basis, grid)` binds a closed-surface basis to evaluation
points. `synthesize_scalar` and `synthesize_helmholtz` map coefficients to
samples. `analyze_scalar` and `analyze_helmholtz` fit samples already on the
bound grid. `analyze_scalar_samples` and `analyze_helmholtz_samples` accept
external sample grids and return batch-first coefficient rows. By default they
analyze in the transform's basis; an explicit `analysis_basis` selects another
sample-analysis route, but the returned coefficients always belong to the
transform's basis. Spectral bases fit the input samples directly, while cell-centred
cubed-sphere bases remap them to the bound grid before target-basis analysis.
Source-grid `sqrt_weights` cannot be carried through a grid remap; configure
target-grid weights on the transform instead. `with_basis` returns a transform
for another coefficient basis while reusing the same grid and numerical policy.

Matrix names state what they do and end in `_matrix`; structured equivalents
end in `_operator`. For example, `scalar_synthesis_matrix` and
`helmholtz_synthesis_operator` have the same semantic direction. Geometric
component transformations spell out both directions instead of changing
meaning through an `inverse` flag.

The word *projection* in Kompe's type vocabulary refers to coordinate charts.
Mean-free coefficient projections retain that mathematical name because they
are explicit projections within one coefficient space.

## Numerical layer

`kompe.math.LinearMap` is the common structured-operator contract. Use
`as_linear_map` to wrap dense or sparse arrays and the named constructors for
diagonal, identity, pointwise, stacked, or indexed maps. Least-squares helpers
consume `LinearMap` without requiring dense materialization.
Custom matrix-free maps use the ordinary constructor keywords `matvec`,
`rmatvec`, and optional `matmat`, `rmatmat`, `dense_array`, or `diagonal`
functions; implementation storage remains private.
Materializing with `to_array()` or `to_matrix()` caches the dense matrix; later
applications reuse it instead of repeating the structured construction.
Known diagonal and identity maps remain vector-backed even when a full matrix
has been requested for inspection.

NumPy/SciPy is the reference backend. JAX is optional and lazy. Select it with
`kompe.math.set_backend("jax")`, a `backend_context("jax")`, or explicit JAX
arrays. Kompe never changes JAX's global 64-bit setting.

## Cache lifecycle

`GlobalCSBasis` and `SphericalTransform` expose `cache_info()` and
`clear_cache()`. Caches use exact representation/grid signatures; shared CS
remap matrices and per-basis target-grid caches are bounded. Long-running
applications can clear them at known lifecycle boundaries without changing
numerical results.
