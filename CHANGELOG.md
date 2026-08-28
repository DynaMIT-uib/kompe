# Changelog

All notable changes to Kompe will be recorded here.

The project follows semantic versioning. Until 1.0, minor releases may refine
public interfaces with an accompanying migration note.

## Unreleased

- Give matrix-free `LinearMap` construction public operation keywords, and
  distinguish normalized solid-harmonic potential-jump factors from the
  dimensioned jump at a specified radius.
- Make local tests import the checkout under test, cover Python 3.14, and keep
  scheduled downstream compatibility testing focused on PynaMIT.
- Preserve the caller's broadcast shape on `SphericalGrid` while keeping
  coordinates flat for numerical operators.
- Clarify spherical coordinate frames, regional centre tuple order, area
  weighting, and scaled regularization in the public scientific API.
- Reuse an explicitly materialized `LinearMap` matrix for later applications,
  avoiding repeated structured construction after the dense cost has been paid.
- Preserve that dense materialization when the same `LinearMap` is relabeled
  with shaped input/output metadata.
- Keep known diagonal and identity maps on their vector-backed application path
  even when a full matrix has been requested for inspection.
- Make the reusable dense normal-pseudoinverse solver retain an explicit data
  adjoint, keeping its repeated solve path fast without changing `LinearMap` semantics.
- Remove the empty `SphericalBasis` and `SphericalRepresentation` aliases;
  `ScalarBasis` is the direct coefficient-basis interface.
- Remove coefficient-space aliases from `SphericalGrid`; compare point grids
  directly with `same_as()`.
- Use Python's standard cached-property mechanism for lazy spherical-transform
  matrices, operators, and least-squares problems while retaining `clear_cache()`.
- Keep sample-analysis branches beside the workflow they control, and dispatch
  fixed least-squares solvers and preconditioners directly instead of through
  single-use helpers and a per-instance method registry.
- Use ordinary ``LinearMap`` composition for synthesis products instead of a
  transform-specific matrix-composition wrapper.
- Store concrete representation metadata as ordinary validated attributes
  instead of repeating abstract-property boilerplate in every grid and basis.
- Reduce spherical-harmonic index setup to the coefficient pairs and arrays it
  actually uses, and reuse the degree/order lookup during recurrence evaluation.
- Remove unused cubed-sphere and backend helpers.
- Remove the dynamic `xp` namespace proxy; numerical code now selects NumPy or
  JAX explicitly from its operands with `get_array_module()`.
- Make the surface-basis mean-free contract explicit instead of letting
  consumers silently skip a missing gauge projection.
- Use the canonical `(theta, phi)` component order for regional gradient and
  divergence operators, matching every other Kompe surface operator; expose
  `(xi, eta)` partial derivatives under their own explicit method.
- Use canonical `(theta, phi, radial)` components for global-CS vector
  interpolation and expose physical ENU-to-cube transforms by direction,
  instead of public unnormalized component matrices plus an inverse flag.
- Keep regional dual-basis geometry private and cached, and remove unused
  component-order metadata and spherical tangent code from Kompe.
- Name regional physical resolution by direction with `xi_cell_size` and
  `eta_cell_size`; retain the tuple form only at the version-1
  serialization/compatibility boundary.

## 0.2.0

- Establish a uniform public vocabulary: projections own continuous charts,
  meshes own cell geometry, bases define expansion spaces, transforms perform
  analysis/synthesis, and operators represent linear actions.
- Rename `Grid`, `RegionalCSGrid`, `SurfaceOperators`, `SolidHarmonics`, and
  related methods to their descriptive canonical forms.
- Add `GlobalCSProjection` and separate global mesh geometry from
  `GlobalCSBasis`.
- Make regional mesh construction explicit through exactly one of `shape`,
  explicit `xi_cell_size` and `eta_cell_size` values, or exact edge arrays.
- Keep historical consumer names in PynaMIT and secsy rather than adding
  compatibility aliases to Kompe.

## 0.1.0

- Establish Kompe as a standalone spherical-computation package.
- Add first-class SH, SECS, global cubed-sphere, and regional cubed-sphere
  representations.
- Separate regional grid geometry from `RegionalCSOperators`.
- Add the versioned `RegionalCSMeshSpec` interchange format.
- Provide backend-neutral linear maps and least-squares machinery, with
  optional JAX support.
- Load JAX lazily without changing process-wide precision configuration.
- Use analysis terminology for coefficient fitting and reserve projection
  objects for coordinate charts.
- Require explicit global-CS resolution and SECS current-system mode.
- Add bounded cache controls, release gates, provenance, benchmarks, and
  standalone transform coverage.
