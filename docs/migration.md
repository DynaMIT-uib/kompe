# Consumer migration

## PynaMIT

PynaMIT depends on `kompe` directly. Spherical and numerical code should be
imported from its owning package:

```python
import kompe
basis = kompe.GlobalCSBasis(N=16)
```

The sole retained PynaMIT spelling is
`pynamit.BasisEvaluator`, an alias of `kompe.SphericalTransform`. There is no
general `pynamit.math` facade and no general spherical re-export surface.
`KOMPE_USE_JAX` and `KOMPE_LEAST_SQUARES_SOLVER` are the only numerical
environment settings.

## Lompe

Lompe imports its regional cubed-sphere and SECS matrices from `kompe`.
Its working grid remains the structured `RegionalCSGrid`; no global basis is
required for the current inversion.

Lompe does not maintain a parallel grid class or a legacy `lompe.cs` namespace.
Its input adapter reconstructs historical grid-like objects and saved grid
dictionaries as canonical `RegionalCSGrid` instances before the model uses
them.

New saved grids use the versioned `RegionalCSGridSpec.to_mapping()` format.
The adapter continues to accept historical `L`/`W`/`Lres`/`Wres` dictionaries,
but that vocabulary is not part of Kompe's current public serialization API.
Differential and interpolation operations are accessed through
`grid.operators`. Canonical construction uses descriptive names and requires
an explicit radius, for example
`RegionalCSGrid(projection, length, width, length_resolution,
width_resolution, radius=radius)`.

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
`RegionalCSGrid` instances use the uniform `kompe` contract: canonical
angular coordinates are degrees, and `_rad` accessors are explicit.
Legacy `get_SECS_*` names and `CSgrid.get_Le_Ln()` remain implemented in secsy
as thin translations to canonical Kompe kernels and `RegionalCSOperators`.

## Representation choice

```text
SphericalRepresentation
├── Grid                               unstructured sample points
├── RegionalCSGrid                     structured regional mesh values
└── SphericalBasis
    └── ScalarSynthesis
        ├── SECSBasis                  Green-function/current synthesis
        └── SurfaceOperators
            ├── SHBasis                global spectral basis
            └── GlobalCSBasis          global six-face nodal basis

StructuredSurfaceMesh
├── RegionalCSGrid                     bounded single-face mesh
└── GlobalCSMesh                       closed six-face native mesh
    └── composed by GlobalCSBasis.mesh
```

`SECSBasis` deliberately does not inherit `SurfaceOperators`: its Green
functions have distributional Laplacians at their poles, so pretending it has
the same square coefficient-space Laplacian and mean-free Poisson semantics as
SH or global CS would make the interface less honest. It is nevertheless a
first-class representation with signatures, scalar synthesis, canonical
surface-current operators, two-component Helmholtz synthesis, and magnetic
field synthesis.
