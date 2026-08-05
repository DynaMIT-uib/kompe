# Architecture

The dependency rule is strict: `kompe` may depend on general numerical
libraries, but never on PynaMIT, Lompe, or secsy. Consumer-specific wrappers
belong to those consumers or to explicitly named compatibility modules.

## Representation hierarchy

- `SphericalRepresentation` owns validated coefficient or sample metadata.
- `SphericalBasis` identifies reconstructive representations.
- `ScalarBasis` adds scalar evaluation on a target grid.
- `SurfaceDifferentialBasis` adds closed-surface gradient, Helmholtz, Laplacian, and
  gauge-aware Poisson capabilities.
- `SECSBasis` is a `SphericalBasis` with scalar-potential, surface-current,
  and magnetic-field synthesis. Its required `current_type` gives one
  coefficient vector an explicit curl-free or divergence-free meaning. Its
  Green functions are distributional, so it does not pretend to have a square
  coefficient-space Laplacian.
- `SphericalGrid` stores arbitrary evaluation points and does not imply
  cells or topology.
- `StructuredSurfaceMesh` describes cell-centred structured surface geometry.
  `RegionalCSMesh` and `GlobalCSMesh` implement it.
- `RegionalCSMesh` is bounded mesh geometry. Its composed
  `RegionalCSOperators` owns interpolation, gradient, metric, and divergence
  operations. Boundary stencils act on that mesh; it is not a closed-surface
  basis.

## Operators, meshes, projections, and expansions

The expanded project name is also its separation rule:

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

## Cubed-sphere ownership

Stateless gnomonic coordinates, metrics, and vector transformations are shared.
`GlobalCSMesh` owns immutable six-face cell geometry, while `GlobalCSBasis`
owns the expansion, cross-face stencils, remapping, and closed-sphere
operators. `RegionalCSMesh` owns the rotated single-face mesh, while
`RegionalCSOperators` composes that geometry into interpolation and
differential operators for a bounded patch. `RegionalCSMeshSpec` provides the
versioned JSON boundary. Consumer-specific object translation remains outside
Kompe.

## Numerical layer

`kompe.math` owns `LinearMap`, backend selection, fingerprints, tensor
operations, and least-squares factorization. Consumers import those objects
directly, ensuring there is only one implementation and one class identity.
Persistent caches are accepted through the small `get_or_create` protocol and
remain consumer-owned.

JAX is an optional execution backend, not an import-time policy. Merely
importing Kompe does not import JAX or enable 64-bit values. Explicit JAX
arrays select JAX for compatible operations, and callers can select the global
default with `set_backend("jax")`. NumPy/SciPy remain the reference path;
SciPy-only sparse formats may stay on that path when no equivalent JAX
operation exists.

## Component frames

Canonical surface vectors use `(theta, phi)`, equivalent to `(south, east)`.
The conversion to geographic EN components is `east = phi`, `north = -theta`.
Full magnetic vectors use `(radial, theta, phi)`. SECS kernels use geographic
`(east, north, radial)` return order; secsy retains its historical names in its
own facade.
