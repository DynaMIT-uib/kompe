# Consumer migration

## Kompe operator names

`LinearMap` construction now uses the public operation names `matvec`,
`rmatvec`, `matmat`, and `rmatmat` instead of private field keywords.
Dense shaped materialization is now explicit through `to_array()`; the former
`.array` property is removed. `to_matrix()` continues to return the flat 2-D
representation. Surface differential bases expose one canonical
`surface_laplacian_operator()`; explicitly materialize it when a matrix is
needed. Helmholtz component, divergence, and curl maps likewise use their
`_operator` methods rather than parallel matrix wrappers.

`SphericalTransform.analyze_scalar()` and `analyze_helmholtz()` accept the
solver name as `solver=`. Transform and sample-analysis tolerances are set
with `tolerance=`; the pseudoinverse-specific `pinv_rtol` name is removed
because the same value also configures iterative solvers. Coordinate
conversions and rotations live in
`kompe.spherical_coordinates` rather than the ambiguous `kompe.spherical`
module.

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

basis = kompe.GlobalCSBasis(cells_per_face=16)
```

PynaMIT has no parallel spherical facade: use `kompe.SphericalTransform`
directly. There is no general `pynamit.math` facade or spherical re-export surface.
`KOMPE_USE_JAX` and `KOMPE_LEAST_SQUARES_SOLVER` are the only numerical
environment settings.

Historical `SphericalTransform.project_scalar` and `project_helmholtz` calls
become `analyze_scalar_samples` and `analyze_helmholtz_samples`; the
`projection_basis` argument becomes `analysis_basis`. This keeps coordinate
projections distinct from coefficient analysis. Use `scalar_synthesis_matrix`
and `helmholtz_synthesis_matrix` for the former short matrix spellings `G` and
`G_helmholtz`.

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

Global-CS direct vector interpolation is now `interpolate_vector()` in
`(theta, phi, radial)` order. The old east/north
`interpolate_vector_components()` entry point is removed. Physical chart
transforms are `enu_to_cube_vector_matrix()` and
`cube_to_enu_vector_matrix()`; the former public unnormalized spherical
component and normalization matrices are now internal implementation details.

Lompe does not own a spherical-harmonic representation or transform. It calls
`ppigrf` for the reference geomagnetic field; that external model is based on
spherical-harmonic coefficients internally, but exposing those implementation
details as a Lompe basis would conflate field-model evaluation with Lompe's
inversion representation. `SHBasis` therefore remains available from
`kompe` without becoming a Lompe dependency boundary.

## secsy

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
first-class basis with signatures, scalar Green-profile synthesis, canonical
surface-current operators, two-component Helmholtz synthesis, and magnetic
field synthesis. Construction requires `current_type="curl_free"` or
`current_type="divergence_free"`; this prevents a scalar coefficient vector
from silently changing physical interpretation.
