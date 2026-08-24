# Kompe

**KOMPE — Kit for Operators, Meshes, Projections, and Expansions**

`kompe` provides regional and global numerical methods for spherical geometry
and fields. Its name comes from the round Norwegian potato dumpling and
deliberately complements the `lompe` package it serves.

The four terms in the name describe separate architectural roles:

- **projections** parameterize continuous spherical geometry and transform
  coordinates and vector components;
- **meshes** discretize that geometry into structured cells with areas,
  boundaries, and implicit neighbourhood topology;
- **expansions** represent fields through bases and coefficients;
- **operators** act between field, sample, and coefficient spaces.

The public implementations include:

- spherical-harmonic (`SHBasis`), Spherical Elementary Current System
  (`SECSBasis`), and global cubed-sphere (`GlobalCSBasis`) expansions;
- regional and global cubed-sphere projections (`RegionalCSProjection` and
  `GlobalCSProjection`);
- regional and global structured cubed-sphere meshes (`RegionalCSMesh` and
  `GlobalCSMesh`);
- scalar, tangential, magnetic-field, analysis, synthesis, differential, and
  transfer operators;
- backend-neutral `LinearMap` objects and least-squares solvers.

The package depends only on NumPy and SciPy. JAX support is
optional and loaded only when requested; Kompe does not change JAX's global
precision configuration. It never imports PynaMIT, Lompe, or secsy; those
libraries are consumers or compatibility facades.

## Dependency direction

```text
                 kompe
              /      |      \
         PynaMIT    Lompe    secsy compatibility API
```

PynaMIT imports Kompe directly. Legacy secsy function names and class spellings
remain in secsy rather than becoming aliases in Kompe. Lompe consumes the
canonical regional mesh directly and translates historical serialized grids
at its own package boundary.

## Geometry and representation boundaries

`SphericalGrid` is an arbitrary set of
evaluation or observation points; it does not claim cells or topology.
`StructuredSurfaceMesh` is the distinct contract implemented by the regional
CS mesh and the native `GlobalCSMesh` exposed as `GlobalCSBasis.mesh`.
`RegionalCSMesh` owns geometry; its cached `operators` object owns gradient,
divergence, surface-metric, and interpolation operations. A versioned
`RegionalCSMeshSpec` is the stable interchange format for saved grids and
consumer translation layers.

An expansion does not have to use a mesh. SH and SECS expansions are analytic
or kernel-based, while the global CS expansion is supported by its native
mesh. Analysis and synthesis are operators associated with expansions rather
than alternate names for coefficient fitting. In the four-part architecture,
Projection objects are geometric coordinate charts and their vector/Jacobian
transformations; coefficient fitting is called analysis.

## Conventions

All public coordinate and component conventions are explicit:

- `SphericalGrid` latitude/longitude and canonical `theta`/`phi` are degrees in
  the spherical frame chosen by the caller;
- cubed-sphere `xi`/`eta` are radians;
- radii are unit-agnostic but must be mutually consistent;
- surface-operator components are `(theta, phi)` (south, east);
- SECS kernels return `(east, north[, radial])`.

Regional meshes are bounded patches. They share geometry and operator
capabilities with closed-sphere representations, but do not claim
closed-surface Helmholtz or mean-free Poisson semantics.

## Quick start

```python
from kompe import RegionalCSMesh, RegionalCSProjection

centre_longitude = 20.0
centre_latitude = 70.0
projection = RegionalCSProjection((centre_longitude, centre_latitude), orientation=0.0)
mesh = RegionalCSMesh(
    projection,
    length=1800.0,
    width=1400.0,
    radius=6371.2,
    shape=(14, 18),
)

theta_gradient, phi_gradient = mesh.operators.surface_gradient_matrices(sparse=True)
divergence = mesh.operators.surface_divergence_matrix(sparse=True)

metadata = mesh.to_spec().to_dict()
restored = RegionalCSMesh.from_spec(metadata)
assert restored.signature == mesh.signature
```

For global basis fitting and evaluation, the workflow is equally direct:

```python
import numpy as np
from kompe import SHBasis, SphericalGrid, SphericalTransform

latitude = np.linspace(-87.5, 87.5, 36)
longitude = np.linspace(-180.0, 175.0, 72)
lon, lat = np.meshgrid(longitude, latitude)
grid = SphericalGrid(lat=lat, lon=lon)

basis = SHBasis(8, 8, mean_free=False)
transform = SphericalTransform(basis, grid, area_weighted=True)
samples = np.cos(np.deg2rad(lat)).reshape(-1)
coefficients = transform.analyze_scalar(samples)
fitted = transform.synthesize_scalar(coefficients).reshape(grid.shape)
```

When physical resolution is more natural than a cell count, name the two
directions explicitly: `xi_cell_size` is parallel to the projection orientation
and `eta_cell_size` is perpendicular to it. This avoids reversing the physical
axes to match the array shape's `(eta, xi)` order.

Install the numerical core with `pip install kompe`. JAX acceleration is an
optional extra: `pip install "kompe[jax]"`. Select it with
`KOMPE_USE_JAX=1` or `kompe.math.set_backend("jax")`. Applications that need
64-bit JAX arithmetic should set `JAX_ENABLE_X64=1` before importing JAX.

The public API, coordinate conventions, release policy, and consumer migration
are documented in [`docs/`](docs/). Kompe is currently an alpha API: releases
follow semantic versioning, but breaking corrections may occur before 1.0 and
will be recorded in the changelog.
