# Consumer migration

## Scalar transforms and evaluation caching

`SphericalTransform` now accepts `ScalarBasis` for scalar synthesis/fitting,
including SECS potentials. It does not grant those bases closed-surface Helmholtz
or surface-smoothness semantics. `SurfaceDifferentialBasis` is still required
for those operations; arbitrary scalar bases need no new base class.

Custom basis evaluation overrides should accept keyword-only `persist=True`
and forward it to any delegated evaluation. `persist=False` disables evaluation
disk-cache access, not bounded in-memory reuse. The transform's existing
`use_persistent_evaluation_cache=False` now covers vector evaluations too.
The private `_uncached_scalar_evaluation_operator` hook is removed.

SH gradient/rotated-gradient arrays are now optional materializations. Building
a Helmholtz map does not allocate them. Requested arrays and matrices remain
reusable through `LinearMap`; numerical conventions and solver choices are unchanged.

## Fixed analysis gauges and structured prepared fits

Fixed Helmholtz analysis now enforces the same physical zero-mean gauges as
`analyze_helmholtz()`, including non-native CS grids and complete coefficient
subsets. For an untruncated fit, the gauge correction changes only constant
potential offsets, not the reconstructed field. Spectral truncation now acts
in the same constrained coordinates as configurable analysis.

Prepared `normal_pinv` fits no longer materialize the data adjoint simply to
apply it repeatedly. Sparse and matrix-free maps retain their actions; an
already materialized map still uses its cached array. The normal inverse and
solver tolerance retain their previous meaning.

## Constrained fits and iterative initial guesses

`restrict_solution(Z)` now retains only the remaining independent constraints,
including when Z is an independently constructed null-space map. Constraint
row units no longer affect the rank check. Fit tolerances remain unchanged.

Iterative `x0` is in the original coefficient space and supports the same
leading scientific / trailing batch axes as results. Right-preconditioned
LSMR now preserves the initial nullspace component rather than rescaling it.
JAX CGLS warns when its final normal residual misses the requested tolerance.

## Fixed fit inputs, constraints, and SECS potentials

- Problems and transforms own their array-valued residual weights. Construct a
  new fit to change its objective; editing the originally supplied NumPy array
  no longer changes an existing fit. Matrix-free operator actions must remain
  fixed for the lifetime of their problem.
- Use `LeastSquaresProblem(..., constraints=C)` for exact `C x = 0` conditions.
  The solver handles sparse KKT or orthonormal independent coordinates as
  appropriate to the selected algorithm. Solutions retain the original shaped
  coefficient space. The independent map is `problem.solution_basis`, not
  `transform.helmholtz_gauge_basis`.
- `sparse_least_squares_map` replaces `sparse_constrained_least_squares_map`;
  constraints are optional, and `regularization=R` adds an already-scaled
  sparse penalty. Both names describe fixed response maps, not iterative solves.
- SECS scalar synthesis now always evaluates a current potential: Phi for
  `J_cf = -grad_s(Phi)` or Psi for `J_df = rhat x grad_s(Psi)`. The old DF scalar
  magnitude-profile behavior was not a potential or the magnitude of a summed
  current. Low-level `scalar_green_matrix` requires an explicit quantity:
  `curl_free_potential`, `divergence_free_potential`, or `current_profile`.
  The last is the angular cotangent profile before radius/direction factors.
  SECS scalar gradient components now follow the ordinary unit-sphere/radian
  derivative contract; surface-current and magnetic-field synthesis are unchanged.

## Basis operators, smoothness, and coefficient batches

- Custom scalar bases implement `scalar_evaluation_operator`, returning a
  `LinearMap`. They no longer need to implement an explicit array builder.
  `scalar_evaluation_array` remains available as materialization. SH's direct
  analytical arrays and diagonal coefficient Laplacian are unchanged.
- A proper subset of CS coefficients no longer silently crops the output of
  `surface_laplacian_operator`. Use `laplacian_evaluation_operator(grid, r)`
  for derivative values; compose a projection explicitly if a smaller-space
  approximation is intended. Gradient evaluation still requires no fitting.
- Replace `scalar_smoothness_weights()` and `helmholtz_smoothness_weights()`
  with the corresponding `*_smoothness_operator()` methods. Use `.diagonal()`
  when specifically inspecting SH modal weights. CS penalties instead retain
  sparse derivative residuals; generic regularization code consumes the map.
- Coefficient normalization now uses the same layout as analysis and synthesis:
  `(n_coeffs, *batch)` or `(2, n_coeffs, *batch)`. Flattened field axes are also
  accepted, **before** batch axes. `scalar_mean` contracts the first axis.
  Convert time-first arrays with `xp.moveaxis(values, 0, -1)` at storage or
  provider boundaries. `FieldCoefficients` has been removed; use ordinary arrays
  and `CoefficientSpace.project_mean_free` for explicit shape/gauge normalization.
  Stored rows already normalized by a consumer need no additional wrapper.
- A custom `LinearMap(normal_matrix_diag=callback)` callback accepts optional
  `row_scale=None` to support weighted normal diagonals without probing the
  operator. The result is `diag(A* diag(abs(row_scale)**2) A)`, or `diag(A* A)`
  when omitted. Sparse, diagonal, component, selection and einsum factories
  supply these reductions themselves.

## Explicit integration and regularization; CS ownership

- `area_weighted=True` now requires supplied `grid.area_weights` or explicit
  `sqrt_weights`. Replace the previous inferred `sin(theta)` only where the
  sampling really is uniform in theta/phi, by attaching those weights to the
  grid. Equal-area point sets need constant weights. Native CS grids already
  carry their cell areas.
- `GlobalCSMesh.operators` owns native differential calculations, shared by
  bases constructed with that mesh. Basis evaluation remains common to SH/CS.
- Geometric interpolation moved off `GlobalCSBasis`: construct
  `GlobalCSRemapper()` directly, or reuse `cs_basis.remapper`. Its
  `scalar_operator(source, target)` and `tangential_operator(source, target)`
  replace the basis's grid-remap factories; `interpolate_scalar/vector`
  provide the direct SciPy interpolation calls.
- `LeastSquaresProblem` takes `regularization=R`, where R is already scaled.
  `regularization_operators=` and `regularization_strengths=` constructor
  arguments are removed. To retain the former relative objective, construct
  `R = relative_regularization(A, L, strength, sqrt_weights=weights)` and then
  `LeastSquaresProblem(A, sqrt_weights=weights, regularization=R)`. For several
  data terms, balance against their stacked weighted operator. For several
  penalties, pass a list of scaled maps. `reg_lambda` on spherical transforms
  keeps its existing relative meaning; do not rescale it during migration.

## Function locations

- The standalone `kompe.mesh.spherical_triangle_solid_angle` replaces
  `GlobalCSMesh.spherical_triangle_area`. Inputs are unit Cartesian vertices,
  with broadcast leading axes; results are unsigned solid angles in steradians.
- Import `finite_difference_weights`, `determinant_3x3`, and `inverse_3x3`
  from `kompe.math`. Their implementations are in `math.finite_differences`
  and `math.small_matrices`, not `cubed_sphere`.
- Schmidt normalization factors moved from
  `spherical_harmonics.coefficients` to `spherical_harmonics.normalization`.
- The private basis helper `_owned_readonly_array` is now
  `kompe.math.readonly_numpy_array`; it owns a read-only contiguous CPU copy.

## Kompe operator names

`tensor_pinv` and `weighted_tensor_pinv` now use `output_ndim` for the split
between output and input axes, replacing `n_leading_flattened`. Its default
is **1**, so an ordinary matrix gets its ordinary pseudoinverse. For arrays
with two output axes, such as `(2, n_points, n_coefficients)`, explicitly pass
`output_ndim=2`. These functions invert one tensor-shaped map, not a batch
of independent matrices. Kompe requires NumPy 2.0 or later (including its
`linalg.pinv(rtol=...)` interface).

`LeastSquaresProblem` now takes shapes only from its data operators; remove
`solution_shape=` and `data_shapes=`. Raw matrices need no replacement arguments.
For scientific axes, construct
`A = as_linear_map(array, input_shape=..., output_shape=...)`, then use
`LeastSquaresProblem(A, ...)`. Options after `A` are keyword-only.

`LeastSquaresProblem.A` is now `data_operators`; its former `sqrt_weights`
attribute is `weight_operators`, because it stores maps rather than the raw
constructor weights. The constructor keywords `A` and `sqrt_weights` are
unchanged. Use `len(problem.data_operators)` or
`len(problem.regularization_operators)` instead of stored term counts.

`LeastSquaresProblem.svd` is now a method: use `problem.svd()` or select
`backend="numpy"` / `backend="jax"` explicitly. It retains one reduced
factorization per backend rather than forcing all SVD work onto the CPU.

Dense `LeastSquaresSolver` algorithms now reject unsupported keyword options
such as `damp` and `maxiter`; these were previously ignored. LSMR also rejects
nonzero `damp` with a right preconditioner, which previously changed the
regularization coordinates. Put the penalty explicitly in the problem when
preconditioning a regularized solve.

`BasisSubset.with_mean_free()` returns itself when its intrinsic mean status
already matches, or an explicitly registered related space. The SH
full/mean-free pair continues to round-trip. An arbitrary subset cannot infer
a different gauge and raises `NotImplementedError` instead of silently
returning a larger parent-derived coefficient space.

`basis.mean_free` and the default `CoefficientSpace.mean_free` now describe
zero surface means, not merely the absence of a representable constant.
For example, dropping one CS node removes the constant but does not make
the remaining fields mean-free. Explicit mean-free projection on such a
subset raises an error because a constant shift is outside the space;
use a constrained fit instead. Full SH bases now remove their monopole
when explicitly asked to project coefficients mean-free.

`LinearMap` is callable: `operator(values)` preserves its declared domain,
codomain, and trailing batch axes. Use this instead of manually flattening
fields for `matvec` and reshaping the result. The flat `@` interface is unchanged.
`matmat` and `rmatmat` now consistently return a 1-D vector for 1-D input;
use `(n, 1)` input for a one-column block. Blocks must be 2-D with the expected
row count; use `operator(values)` for scientific arrays with additional axes.
All scalar and Helmholtz analysis/synthesis methods now retain trailing batch
axes, including `analyze_scalar_samples` and `analyze_helmholtz_samples`.
Scalar input `(points, *batch)` returns `(coefficients, *batch)`; tangential
input `(2, points, *batch)` returns `(2, coefficients, *batch)`. Unbatched
inputs no longer gain a length-one time axis. Least-squares right-hand sides
also require leading data axes. Convert time-first arrays with
`xp.moveaxis(values, 0, -1)` at your input boundary and move the last axis
back only when storing a time series. Flatten shaped spatial axes explicitly.
The transform's time-row conversion methods have been removed.

Sample analysis no longer accepts `analysis_basis` or dispatches on
`sample_analysis_uses_grid_remapping`. Its coefficient basis is always
`transform.basis`. To remap before fitting, pass a scalar or tangential
`LinearMap` as `remap=`, for example one returned by
`cs_basis.remapper.tangential_operator(input_grid, transform.grid)`.
Without a map, samples are fitted directly on `input_grid`. Use `with_basis`
when intentionally changing the fitted representation or evaluation algorithm.

`theta_derivative_array/operator` and `phi_derivative_array/operator` are now
`gradient_theta_array/operator` and `gradient_phi_array/operator`.
The `derivative=` keyword in scalar evaluation/synthesis is now
`gradient_component=`: its phi component includes `1/sin(theta)` and has
always been a unit-sphere gradient component, not a bare partial derivative.
The numerical values and cache identities are unchanged.

`cholesky_least_squares_map(synthesis, normal_factor, sqrt_weights=...)`
infers the analysis axes by reversing the synthesis map's axes. It no longer
accepts redundant `input_shape` or `output_shape` arguments. Label the
synthesis operator once with `as_linear_map` when needed. The dense
full-rank convenience factory uses this same forward/adjoint implementation.
`dense_full_rank_least_squares_factor` is removed: use
`dense_full_rank_least_squares_map(A)` to build and retain the factor, or
`cholesky_least_squares_map(A, factor)` when a factor is already available.

`SphericalTransform.cache_info()` uses `cached_attributes` and
`scalar_problem_cached`/`helmholtz_problem_cached` in place of
`materialized_values` and `scalar_factorization`/`helmholtz_factorization`.
Constructing a problem is not itself a factorization.
Helmholtz sample analysis keeps the two potentials on a distinct axis as
`(2, n_coefficients, *batch)`, rather than flattening potentials into coefficients.

`LinearMap.diagonal()` requires declared diagonal structure. It no longer
materializes an unknown operator to test whether its off-diagonal entries
happen to be zero. Use `diagonal_linear_map` when constructing a diagonal map.

`take_linear_map(shape, index, axis=...)` removes the selected axis when
`index` is a scalar integer; use `[index]` to retain a length-one axis.
Index arrays and axis masks retain the axis, and repeated indices add in the
adjoint. Indices must be nonnegative integers: floats are no longer silently
truncated. Maps own their index metadata, so later caller edits cannot change
the selection. Helmholtz component operators now use this same primitive.

Custom `ScalarBasis` implementations must define `coefficient_space_signature`.
Override `signature` as well when evaluation choices affect cached arrays
without changing the coefficient layout.

`LinearMap` construction now uses the public operation names `matvec`,
`rmatvec`, `matmat`, and `rmatmat` instead of private field keywords.
Dense shaped materialization is now explicit through `to_array()`; the former
`.array` property is removed. `to_matrix()` continues to return the flat 2-D
representation. Surface differential bases expose one canonical
`surface_laplacian_operator()`; call `to_matrix()` when its flat matrix is
needed. Helmholtz component, divergence, and curl maps likewise use their
`_operator` methods rather than parallel matrix wrappers.

All `SphericalTransform.analyze_*` methods accept a solver name or a reusable
`LeastSquaresSolver` as `solver=`. Configure a shared tolerance on that solver,
or set the transform's default with `SphericalTransform(..., tolerance=...)`.
The sample methods no longer take a separate tolerance: child transforms
inherit the parent default rather than resetting it. The old `pinv_rtol`
name is removed. Fits no longer substitute a direct inverse for an iterative
or pseudoinverse algorithm. Selecting `normal_solve` does reuse an available
Cholesky/sparse-LU factorization of the same unregularized normal equations.

`LeastSquaresSolver` selects its algorithm with `method=`, not `solver=`;
the corresponding attribute is `.method`. Transform analysis still accepts
the solver object as `solver=`. `LeastSquaresSolver()` and
`LeastSquaresSolver(method=None)` use the same
`KOMPE_LEAST_SQUARES_SOLVER` default as transform analysis (`normal_pinv` when
unset), rather than independently defaulting to LSMR. Request `"lsmr"` explicitly
when that algorithm is intended. Existing solver objects keep their selected
algorithm if the environment changes.

Sample analysis now inherits the transform's `reg_lambda` when omitted or
`None`, consistently for direct and remapped fits. Pass `reg_lambda=0` to
request an unregularized fit. Direct fits on the bound grid also inherit its
explicit `sqrt_weights`; pass unit weights to request an unweighted fit.
Explicit bound-grid weights are not reused on a different input grid.
Request `helmholtz_analysis_operator` explicitly for a fixed, reusable geometric
inverse independent of the selected fit algorithm. Helmholtz gauges now use
`LeastSquaresProblem(constraints=...)`; augmented gauge
penalties and `LeastSquaresProblem(gauge_operator=...)` have been removed.
Coordinate
conversions and rotations live in
`kompe.spherical_coordinates` rather than the ambiguous `kompe.spherical`
module.

`spherical_to_cartesian()` and `cartesian_to_spherical()` now broadcast their
coordinate inputs and return `(3, *broadcast_shape)` arrays. A scalar point
therefore returns shape `(3,)`, rather than `(3, 1)`; multidimensional sample
axes are preserved instead of being folded into the component axis. Geographic
rotations likewise accept broadcast coordinates and ordinary lists on both
NumPy and JAX.

`SolidHarmonicOperators` distinguishes dimensionless conversion factors from
the physical potential jump at a radius. Use
`poloidal_to_regular_potential_factors`,
`poloidal_to_irregular_potential_factors`, and
`poloidal_to_normalized_potential_jump_factors` for coefficient-space factors;
use `poloidal_to_potential_jump_factors(radius)` or
`poloidal_to_potential_jump_operator(radius)` for the dimensioned jump.

## PynaMIT

PynaMIT depends on `kompe` directly. Spherical and numerical code should be
imported from its owning package:

```python
import kompe

basis = kompe.GlobalCSBasis(cells_per_edge=16)
```

PynaMIT has no parallel spherical facade: use `kompe.SphericalTransform`
directly. There is no general `pynamit.math` facade or spherical re-export surface.
`KOMPE_USE_JAX` and `KOMPE_LEAST_SQUARES_SOLVER` are the only numerical
environment settings.

Historical `SphericalTransform.project_scalar` and `project_helmholtz` calls
become `analyze_scalar_samples` and `analyze_helmholtz_samples`, using the
canonical layouts and explicit `remap` described above. Coordinate projections
remain distinct from coefficient analysis. Use `scalar_synthesis_array`
for the former scalar `G` spelling and `helmholtz_synthesis_array` for the
former shaped `G_helmholtz` spelling.

## Lompe

Lompe imports its regional cubed-sphere and SECS matrices from `kompe`.
Its working mesh remains the structured `RegionalCSMesh`; no global basis is
required for the current inversion.

Lompe does not maintain a parallel grid class or a legacy `lompe.cs` namespace.
Its input adapter reconstructs historical grid-like objects and saved grid
dictionaries as canonical `RegionalCSMesh` instances before the model uses
them.

New saved grids use the versioned `RegionalCSMeshSpec.to_dict()` format.
The adapter continues to accept historical `L`/`W`/`Lres`/`Wres` dictionaries,
but that vocabulary is not part of Kompe's current public serialization API.
In those dictionaries explicit `edges` are already the final geometry, matching
the historical secsy behavior, so an accompanying `wshift` is ignored.
Differential and interpolation operations are accessed through
`mesh.operators`. Canonical construction uses descriptive names and requires
an explicit radius, for example
`RegionalCSMesh(projection, length, width, radius=radius,
xi_cell_size=xi_size, eta_cell_size=eta_size)` or an explicit integer
`shape=(n_eta, n_xi)`.

Regional `surface_gradient_*` and `surface_divergence_*` operators now use
Kompe's canonical `(theta, phi)` component order. Older regional code treated
the same arrays as `(east, north)`; convert with `east = phi` and
`north = -theta`. Use `coordinate_derivative_matrices()` for `(xi, eta)`
partials instead of the former `cube_coordinates=True` mode.

Regional coordinate derivatives retain their compact boundary closure:
`stencil_radius` specifies the interior reach, and boundary stencils are
truncated rather than widened inward. Radius 1 is second-order in the interior
and first-order at the endpoints. This default satisfies discrete integration
by parts with trapezoidal weights on the coordinate interval between the end
cell centres; it is not a finite-volume flux rule at the physical cell edges.
The earlier claim of equal interior and boundary accuracy was incorrect.

Global-CS direct vector interpolation is now `interpolate_vector()` in
`(theta, phi, radial)` order. The old east/north
`interpolate_vector_components()` entry point is removed. Physical chart
transforms are `enu_to_cube_vector_array()` and
`cube_to_enu_vector_array()`; the former public unnormalized spherical
component and normalization matrices are now internal implementation details.

Lompe does not own a spherical-harmonic representation or transform. It calls
`ppigrf` for the reference geomagnetic field; that external model is based on
spherical-harmonic coefficients internally, but exposing those implementation
details as a Lompe basis would conflate field-model evaluation with Lompe's
inversion representation. `SHBasis` therefore remains available from
`kompe` without becoming a Lompe dependency boundary.

## secsy

SECS kernels retain small angular separations and evaluate finite on-axis
limits explicitly. A regularized current is zero at its pole, a divergence-free
magnetic field is radial on its axis away from the source sheet, and a curl-free
field is zero below the sheet, including beneath its pole. True unregularized
singularities are unchanged; coordinates are not shifted away from them.

The `secsy` distribution is a compatibility facade depending on `kompe`.
Its function API and names (`CSprojection`, `CSgrid`, `CSplot`) remain
available. Historical `secsy.CSgrid.theta` and `.phi` stay in radians, with
canonical degrees available as `.theta_deg` and `.phi_deg`. New
`RegionalCSMesh` instances use the uniform `kompe` contract: geographic
coordinates are available through the mesh and its degree-valued
`cell_centers`. The radian attributes exist only on secsy's historical façade.
Legacy `get_SECS_*` names and `CSgrid.get_Le_Ln()` remain implemented in secsy
as thin translations to canonical Kompe kernels and `RegionalCSOperators`.

## Representation choice

```text
SphericalGrid                           unstructured sample points

ScalarBasis
├── SECSBasis                           Green-function/current synthesis
└── SurfaceDifferentialBasis
    ├── SHBasis                         global spectral basis
    └── GlobalCSBasis                   global six-face cell-centred nodal basis

StructuredSurfaceMesh
├── RegionalCSMesh                     bounded single-face mesh
└── GlobalCSMesh                       closed six-face native mesh

GlobalCSBasis.mesh ──► GlobalCSMesh
```

`SECSBasis` deliberately does not inherit `SurfaceDifferentialBasis`: its Green
functions have distributional Laplacians at their poles, so pretending it has
the same square coefficient-space Laplacian and mean-free Poisson semantics as
SH or global CS would make the interface less honest. It is nevertheless a
first-class basis with signatures, scalar current-potential synthesis, canonical
surface-current operators, two-component Helmholtz synthesis, and magnetic
field synthesis. Construction requires `current_type="curl_free"` or
`current_type="divergence_free"`; this prevents a scalar coefficient vector
from silently changing physical interpretation.
