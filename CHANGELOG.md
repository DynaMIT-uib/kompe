# Changelog

All notable changes to Kompe will be recorded here.

The project follows semantic versioning. Until 1.0, minor releases may refine
public interfaces with an accompanying migration note.

## Unreleased

- Make `SphericalBasis` a compatibility name for `ScalarBasis` instead of an
  empty extra hierarchy layer; existing imports and `isinstance` checks keep
  working.
- Use Python's standard cached-property mechanism for lazy spherical-transform
  matrices, operators, and least-squares problems while retaining `clear_cache()`.
- Keep sample-analysis branches beside the workflow they control, and dispatch
  fixed least-squares solvers and preconditioners directly instead of through
  single-use helpers and a per-instance method registry.
- Store concrete representation metadata as ordinary validated attributes
  instead of repeating abstract-property boilerplate in every grid and basis.
- Reduce spherical-harmonic index setup to the coefficient pairs and arrays it
  actually uses, and reuse the degree/order lookup during recurrence evaluation.
- Remove unused cubed-sphere and backend helpers.

## 0.2.0

- Establish a uniform public vocabulary: projections own continuous charts,
  meshes own cell geometry, bases define expansion spaces, transforms perform
  analysis/synthesis, and operators represent linear actions.
- Rename `Grid`, `RegionalCSGrid`, `SurfaceOperators`, `SolidHarmonics`, and
  related methods to their descriptive canonical forms.
- Add `GlobalCSProjection` and separate global mesh geometry from
  `GlobalCSBasis`.
- Make regional mesh construction explicit through exactly one of `shape`,
  `cell_size`, or edge arrays.
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
