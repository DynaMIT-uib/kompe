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

SH surface gradients use the analytic limit at each pole, with tangential
components expressed in the direction of the supplied longitude. Nearby points
are evaluated at their actual coordinates without a sine floor or pole shift.
Both `legendre_method="internal"` and `"scipy"` follow the same convention.
The even, cell-centred global cubed-sphere mesh does not sample either pole.

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
An explicitly different evaluation algorithm is retained even when its
coefficient layout is compatible with the previous basis.

Explicit representations that retain scientific axes end in `_array`, even
when a scalar case happens to be 2-D. Flat linear-algebra representations that
are always exactly 2-D end in `_matrix`; structured equivalents end in
`_operator`. For example, `scalar_synthesis_array`,
`helmholtz_synthesis_array`, and `helmholtz_synthesis_operator` state both their
representation and semantic direction. Vectorized geometric component
transforms likewise end in `_array` because sample axes change their rank.
They spell out both directions instead of changing meaning through an
`inverse` flag.

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
`to_matrix()` returns the flat 2-D representation; `to_array()` returns the
same values with shaped domain and codomain axes. Both cache the dense
representation instead of repeating its construction.
Known diagonal and identity maps remain vector-backed even when a full matrix
has been requested for inspection.
`diagonal()` returns the scale vector only when a diagonal representation is
known. It does not materialize a matrix or transfer it to the CPU to discover
structure. Declare diagonal maps with `diagonal_linear_map(values)`; use
`xp.diag(map.to_matrix())` for the diagonal of a general dense matrix.

`LeastSquaresProblem` exposes `data_operators` and `weight_operators` as lists
of maps, separate from the assembled weighted `data_operator` and regularized
`system_operator`. Relative regularization requires a nonzero weighted data
operator and an explicit nonzero regularizer; it never invents a unit scale or
silently discards a positive strength.

`problem.svd(backend=None)` returns the reduced `U, s, Vh` factors and caches
them separately for NumPy and JAX. SVD solves and spectral preconditioners
reuse these same factors; JAX solves do not round-trip through NumPy.

The normal-equation helpers `dense_normal_equations`, `dense_normal_matrix`,
and `dense_normal_pinv` likewise accept `backend=`. Dense solvers infer the
execution backend from all active operators and the right-hand side, even
when only the regularizer is a JAX array. Reusable response solvers keep
separate materializations when subsequent right-hand sides select different
backends without recomputing their prepared factorization. A custom
data-normal-matrix builder remains an explicit CPU
construction boundary; its completed result is transferred to the selected
backend for factorization and application.

`LeastSquaresSolver.solve(..., **options)` accepts options supported by its
selected iterative algorithm. Dense algorithms (`svd`, `normal_solve`, and
`normal_pinv`) reject extra options instead of silently ignoring them.
LSMR supports `damp=sqrt(reg_lambda)` for coefficient-norm regularization
without augmented rows. Nonzero damping cannot be combined with a right
preconditioner: that would penalize the transformed coordinates instead.
Represent the penalty explicitly in `LeastSquaresProblem` for such solves.
Its `regularization_strengths` are relative and scale-balanced; an absolute
penalty `lambda * ||L x||²` can instead be represented by an additional data
operator `sqrt(lambda) * L` with right-hand side `None` (zero).

NumPy/SciPy is the reference backend. JAX is optional and lazy. Select it with
`kompe.math.set_backend("jax")`, a `backend_context("jax")`, or explicit JAX
arrays. Kompe never changes JAX's global 64-bit setting.
`get_array_module(*arrays, backend=None)` uses an explicit backend when given,
otherwise JAX operands take precedence over the configured default. For
example, `get_array_module(values, backend="numpy")` selects NumPy without
changing the global setting; converting the values remains explicit.
`get_backend(*arrays)` reports the corresponding `"numpy"` or `"jax"` name;
with no operands, `get_backend()` reports the configured default.

## Cache lifecycle

`SHBasis`, `GlobalCSBasis`, and `SphericalTransform` expose `cache_info()` and
`clear_cache()`. Caches use exact representation/grid signatures; shared CS
remap matrices and per-basis target-grid caches are bounded. Long-running
applications can clear them at known lifecycle boundaries without changing
numerical results.

Each concrete basis defines `coefficient_space_signature` from its mathematical
layout and normalization. `signature` additionally identifies evaluation
details such as the SH Legendre algorithm. The base class does not inspect
subclass-specific parameter names to guess identity. `root_basis` returns the
underlying basis before any nested coefficient subsets.
