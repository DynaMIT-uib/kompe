# Kompe

**KOMPE — Kit for Operators, Meshes, Projections, and Expansions**

`kompe` provides regional and global numerical methods for spherical geometry
and fields. Its name comes from the round Norwegian potato dumpling and
deliberately complements the `lompe` package it serves.

Install with `pip install kompe`; the core depends only on NumPy and SciPy.

## Fit and evaluate a field

```python
from kompe import SHBasis, SphericalGrid, SphericalTransform
from kompe.math import get_array_module

xp = get_array_module()
lat, lon = xp.meshgrid(xp.linspace(-87.5, 87.5, 36), xp.linspace(-180.0, 175.0, 72), indexing="ij")
# This grid is uniform in latitude and longitude: dA is proportional to cos(lat).
grid = SphericalGrid(lat=lat, lon=lon, area_weights=xp.cos(xp.deg2rad(lat)))
basis = SHBasis(8, 8, mean_free=False)
transform = SphericalTransform(basis, grid, area_weighted=True)

samples = xp.cos(xp.deg2rad(lat)).reshape(-1)
coefficients = transform.analyze_scalar(samples)
fitted = transform.synthesize_scalar(coefficients).reshape(grid.shape)
```

A grid specifies **where**, a basis specifies **how the field is represented**,
and a transform fits or evaluates that representation. Arrays are sufficient;
coefficient containers and cache configuration are optional advanced tools.
`SHBasis` uses spherical harmonics; `GlobalCSBasis(cells_per_edge=16)` uses a
global cubed-sphere mesh. `SECSBasis` represents elementary-current systems.

Numerical arrays keep scientific axes first and batch axes last. Scalar
analysis maps `(points, *batch)` to `(coefficients, *batch)`; Helmholtz analysis
maps `(2, points, *batch)` to `(2, coefficients, *batch)`. Analysis outputs can
be synthesized directly. Flatten spatial grid axes explicitly; analysis and
synthesis do not guess whether an axis is time, component, or position.

For example, several fields can share one cached fit:

```python
fields = xp.stack((samples, 2 * samples), axis=-1)
coefficient_columns = transform.analyze_scalar(fields)
fitted_columns = transform.synthesize_scalar(coefficient_columns)
```

## Regional meshes

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
```

When physical resolution is more natural than a cell count, name the two
directions explicitly: `xi_cell_size` is parallel to the projection orientation
and `eta_cell_size` is perpendicular to it. This avoids reversing the physical
axes to match the array shape's `(eta, xi)` order.

Unlike point grids, meshes own cells and topology. Regional meshes have
boundaries; they do not imply closed-sphere Helmholtz or Poisson conditions.

## Conventions and backends

- `SphericalGrid` angles are degrees in the caller's spherical frame;
  `theta` is colatitude and `phi` is longitude. Cubed-sphere `xi`/`eta` are radians.
- Tangential components are `(theta, phi)` = `(south, east)`;
  SECS kernels use `(east, north[, radial])` at their geographic boundary.
- Basis gradients are unit-sphere gradients: `gradient_phi_operator` includes
  `1/sin(theta)`. Divide by radius for physical spatial derivatives.
  Regional mesh operators already include their mesh radius.
- `LinearMap` preserves diagonal, sparse, or matrix-free structure. Inspect
  scientific axes with `to_array()` or flat linear-algebra axes with `to_matrix()`.

JAX acceleration is an optional extra: `pip install "kompe[jax]"`. Select it with
`KOMPE_USE_JAX=1` or `kompe.math.set_backend("jax")`. Applications that need
64-bit JAX arithmetic should set `JAX_ENABLE_X64=1` before importing JAX.

See the [API guide](docs/api.md) for remapping, solver choices, optional coefficient
containers, and mesh serialization, and [architecture](docs/architecture.md) for
extension points, caching, and ownership. Kompe does not import its consumers
PynaMIT, Lompe, or secsy. Kompe is currently an alpha API: releases
follow semantic versioning, but breaking corrections may occur before 1.0 and
will be recorded in the changelog.
