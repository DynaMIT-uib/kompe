# Changelog

All notable changes to Kompe will be recorded here.

The project follows semantic versioning. Until 1.0, minor releases may refine
public interfaces with an accompanying migration note.

## 0.1.0

- Establish Kompe as a standalone spherical-computation package.
- Add first-class SH, SECS, global cubed-sphere, and regional cubed-sphere
  representations.
- Separate regional grid geometry from `RegionalCSOperators`.
- Add the versioned `RegionalCSGridSpec` interchange format.
- Provide backend-neutral linear maps and least-squares machinery, with
  optional JAX support.
- Load JAX lazily without changing process-wide precision configuration.
- Use analysis terminology for coefficient fitting and reserve projection
  objects for coordinate charts.
- Require explicit global-CS resolution and SECS current-system mode.
- Add bounded cache controls, release gates, provenance, benchmarks, and
  standalone transform coverage.
