# Consumer migration

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
Differential and interpolation operations are accessed through
`mesh.operators`. Canonical construction uses descriptive names and requires
an explicit radius, for example
`RegionalCSMesh(projection, length, width, radius=radius,
cell_size=(eta_size, xi_size))` or an explicit integer
`shape=(n_eta, n_xi)`.

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
    └── GlobalCSBasis                   global six-face nodal basis

StructuredSurfaceMesh
├── RegionalCSMesh                     bounded single-face mesh
└── GlobalCSMesh                       closed six-face native mesh

GlobalCSBasis.mesh ──► GlobalCSMesh
```

`SECSBasis` deliberately does not inherit `SurfaceDifferentialBasis`: its Green
functions have distributional Laplacians at their poles, so pretending it has
the same square coefficient-space Laplacian and mean-free Poisson semantics as
SH or global CS would make the interface less honest. It is nevertheless a
first-class basis with signatures, scalar synthesis, canonical
surface-current operators, two-component Helmholtz synthesis, and magnetic
field synthesis. Construction requires `current_type="curl_free"` or
`current_type="divergence_free"`; this prevents a scalar coefficient vector
from silently changing physical interpretation.
