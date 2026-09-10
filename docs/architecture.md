# Architecture

The dependency rule is strict: `kompe` may depend on general numerical
libraries, but never on PynaMIT, Lompe, or secsy. Consumer-specific wrappers
belong to those consumers or to explicitly named compatibility modules.

## Basis, grid, and mesh roles

- `ScalarBasis` owns validated coefficient metadata and scalar evaluation on
  a target grid. Its primitive is a `LinearMap`; explicit arrays are a
  materialization interface, not a requirement for defining an expansion.
  A basis can also evaluate gradients without claiming a same-space Laplacian.
- `SurfaceDifferentialBasis` adds closed-surface Helmholtz, Laplacian, and
  gauge-aware Poisson capabilities. It also states the mean-free contract:
  harmonic spaces may omit the constant mode by construction, while nodal
  spaces must provide the physical surface-mean weights and projection.
- `SECSBasis` is a `ScalarBasis` with scalar current-potential, surface-current,
  and magnetic-field synthesis. Its required `current_type` gives one
  coefficient vector an explicit curl-free or divergence-free meaning. Its
  Green functions are distributional, so it does not pretend to have a square
  coefficient-space Laplacian.
- `SphericalTransform` accepts any `ScalarBasis` for scalar synthesis and fitting.
  Only Helmholtz operations and surface-smoothness regularization require the
  stronger differential-basis contract.
- `SphericalGrid` stores arbitrary evaluation points. It does not expose
  coefficient metadata, inherit basis behavior, or imply cells or topology.
  Optional `area_weights` define an explicit quadrature measure; latitude
  alone cannot identify how the points sample the sphere.
- `StructuredSurfaceMesh` describes cell-centred structured surface geometry.
  `RegionalCSMesh` and `GlobalCSMesh` implement it.
- `RegionalCSMesh` is bounded mesh geometry. Its composed
  `RegionalCSOperators` owns interpolation, gradient, metric, and divergence
  operations. Boundary stencils act on that mesh; it is not a closed-surface
  basis.

## Operators, meshes, projections, and expansions

These mathematical roles explain the package boundaries:

- **Projection** objects describe continuous geometry: coordinate charts,
  Jacobians, and vector-frame transformations. Coefficient fitting is called
  analysis instead.
- **Mesh** is discrete geometry: cells, measures, boundaries, and structured
  topology. Arbitrary point grids are deliberately not meshes.
- **Expansion** is a coefficient-based field representation. `SHBasis`,
  `SECSBasis`, and `GlobalCSBasis` define expansion families without storing a
  particular coefficient vector.
- **Operator** is a typed action between coefficient, field, or sample spaces.
  Differentiation, interpolation, remapping, analysis, and synthesis all
  belong here.

The categories compose rather than forming one inheritance tree. SH and SECS
expansions do not require meshes; the global CS expansion uses its
`GlobalCSMesh`; and projections supply geometry from which CS meshes are
constructed.

`SphericalTransform` binds a coefficient basis to a sample grid and owns the
reusable fit/evaluation state. External samples use that same coefficient basis;
an optional explicit remapping operator is applied before a bound-grid fit.
Remap construction stays with the geometry that defines interpolation.
Transform analysis does not select routes by inspecting a basis-kind flag.
Numerical data axes lead and batch axes trail throughout analysis and operator
application. Provider and time-series storage layouts are normalized by their
consumers, not by the transform.

Differentiation is not implicitly analysis. Surface gradients evaluate
derivative values directly, including SH derivatives outside the original
scalar span. A same-space Laplacian requires closure; arbitrary CS subsets
instead expose complete sampled Laplacian values. Coefficient restriction
composes an injection with the parent map and never crops derivative residuals.
Smoothness penalties use the same `LinearMap` contract: SH retains modal
diagonals, while CS retains area-weighted sparse gradient/div–curl residuals.
No additional operator or penalty hierarchy is needed.

## Cubed-sphere ownership

Stateless gnomonic coordinates, metrics, and vector transformations are shared.
`RegionalCSProjection` first rotates into a local frame and then uses face 4,
the north face, of that shared chart; it does not implement a second gnomonic
projection. The meshes remain separate because a bounded patch needs boundary
stencils, while a global mesh needs cross-face neighbours.
`GlobalCSMesh` owns immutable six-face cell geometry and its cached
`GlobalCSOperators` owns native cross-face stencils, gradients, Helmholtz
maps, and Laplacian/Poisson operators. `GlobalCSBasis` interprets cell values
as coefficients and composes native operators with target-grid evaluation.
Different bases using the same mesh share its native operators and unit-sphere
Poisson factors. Radius changes scale the inverse without refactorization.
`GlobalCSRemapper` uses the stateless projection chart and source/target point
grids; it needs neither a basis nor a mesh resolution. Its shared matrix cache
identifies the chart and points, not an unrelated coefficient representation.
`RegionalCSMesh` owns the rotated single-face mesh, while
`RegionalCSOperators` composes that geometry into interpolation and
differential operators for a bounded patch. `RegionalCSMeshSpec` provides the
versioned JSON boundary. Consumer-specific object translation remains outside
Kompe.

## Numerical layer

Coefficient values remain ordinary backend arrays. `CoefficientSpace` describes
their layout and gauge; it does not own another value representation. Consumers
establish ownership when arrays become live state or saved histories.

`LinearMap` owns operator action and declared dense, sparse, and diagonal
structure. The einsum module owns contraction fusion and reuse of completed
contractions. Unsupported fusion retains ordinary composition; it never changes
the mathematical operation or probes an operator by materializing it. Separate
vector and block callbacks remain deliberate, so single-field JAX use does not
pay for reshaping through a batch-only implementation.

`kompe.math` owns `LinearMap`, backend selection, fingerprints, tensor
operations, and least-squares factorization. Consumers import those objects
directly, ensuring there is only one implementation and one class identity.
Persistent caches are accepted through the small `get_or_create` protocol and
remain consumer-owned.

General stencil weights live in `math.finite_differences`; the CPU sparse
operator builders use them without involving spherical-coordinate machinery.
`math.small_matrices` contains fused 3-by-3 determinant and inverse formulas,
while the cubed-sphere modules own the metrics and Jacobians that use them.
Read-only CPU metadata ownership lives at the array boundary as
`readonly_numpy_array`, shared by bases, meshes, and cached analysis weights.
Meshes do not import coefficient-basis code just to copy metadata.
`mesh.spherical_triangle_solid_angle` provides shared spherical cell geometry
without requiring a cubed-sphere mesh object.

Component selection belongs to `take_linear_map`, not to a separate Helmholtz
implementation. Scalar indices use basic slicing rather than gathering;
array indices support repeated samples with an accumulating adjoint. Neither
selection requires a materialized matrix.

Schmidt factors live in `spherical_harmonics.normalization`, alongside the SH
implementation. SH-specific Helmholtz normal-matrix assembly remains next to
`SphericalTransform`: its block signs and component weights are part of that
transform, not generic solver policy.
Its optional data-normal builder now returns backend-native arrays. The
least-squares problem owns the literal regularized objective and the `Z* N Z` coordinate
restriction and exact constraints, while `LinearMap` owns structured actions and normal-diagonal
reductions. There is no parallel Gram-matrix object or solver-selection layer.
Relative regularization is an explicitly invoked `relative_regularization`
construction step; it returns an already-scaled `LinearMap`. Generic problems
do not infer penalty scales. Balance before imposing a coordinate restriction
so the physical objective is unchanged. Transform `reg_lambda` retains its
dimensionless, scale-balanced meaning without changing solver choices.

For sparse direct fitting, `LinearMap` retains a lazy SciPy sparse representation
through weighting, stacking, and coefficient selection. The solver factors the
sparse normal/KKT system, including smoothness penalties; it does not first
expand null-space coordinates into a dense rectangular fit. SVD, pseudoinverse,
and iterative methods keep their distinct requested meanings.

Problems own array-valued weights and compiled iterative kernels, as well as
numerical factors. Mutable caller weights cannot change only the RHS of a
cached objective. JAX iteration and independent RHS batching live at the
`math.jax_iterative` backend boundary; there is no global operator registry.

JAX is an optional execution backend, not an import-time policy. Merely
importing Kompe does not import JAX or enable 64-bit values. Explicit JAX
arrays select JAX for compatible operations, and callers can select the global
default with `set_backend("jax")`. NumPy/SciPy remain the reference path;
SciPy-only sparse formats may stay on that path when no equivalent JAX
operation exists.

Explicit backend resolution belongs in `math.backend.get_array_module`,
shared by map materialization and factorization. Least-squares problems own
their backend-specific SVD factors; solvers and spectral preconditioners
reuse them. The preconditioner applies `V @ diag(weights) @ V*` with a direct
kernel to avoid dispatch overhead. Its real spectral weights make the map
self-adjoint, so forward and adjoint calls share that kernel.

Basis evaluation has one persistence policy: `persist=False` forbids evaluation
disk-cache access while allowing ordinary bounded in-memory reuse. The transform's
`use_persistent_evaluation_cache` forwards that choice for scalar and vector maps.
There is no separate uncached evaluation implementation, and the policy is not
part of the mathematical cache key. Clearing a basis cache drops its ownership;
maps already held by a caller remain usable from their retained operands.

SH vector operators retain the theta/phi derivative components. Both Helmholtz
potential gradients are evaluated as one batch, then combined with the explicit
sign/rotation convention. Stacked, rotated, or full Helmholtz arrays are created
only on request and reused by their owning maps afterward. Array-cache entries
for these explicit views reference the same materialization, not another copy.

Normal-equation solvers use the same backend-selection policy as SVD.
Backend selection reads the problem's operand metadata, not its assembled
system: a persisted inverse must remain usable without rebuilding a normal
matrix just to determine where a calculation should run.
Removing zero regularization rows from a right-hand side does not remove
the regularizer from that policy. Reusable normal-pseudo-inverse responses
retain their prepared inverse and data adjoint as `LinearMap` objects. They
reuse device copies without refactorization, even if later solves evict the
problem's shared factor cache.

## Component frames

Canonical surface vectors use `(theta, phi)`, equivalent to `(south, east)`.
The conversion to geographic EN components is `east = phi`, `north = -theta`.
Regional and global surface operators both follow this convention; local
`(xi, eta)` coordinate derivatives are exposed under a distinct name.
Global-CS vector interpolation also uses `(theta, phi, radial)`. Coordinate
charts expose physical `enu_to_cube_vector_array()` and
`cube_to_enu_vector_array()` directions; unnormalized spherical-coordinate
components remain an internal numerical detail.
Full magnetic vectors use `(radial, theta, phi)`. SECS kernels use geographic
`(east, north, radial)` return order; secsy retains its historical names in its
own facade.
