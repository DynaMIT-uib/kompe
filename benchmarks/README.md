# Kompe microbenchmarks

Run `python benchmarks/benchmark_core.py` from an installed checkout. The
script reports construction and operator-materialization timings for small,
representative global CS, regional CS, SH, and SECS problems. It is intended
for before/after comparisons, not as a hardware-independent pass/fail test.

Use the same Python environment, backend, machine load, and command arguments
for both revisions. Use `--backend numpy` or `--backend jax`; set
`JAX_ENABLE_X64=1` for matching double-precision calculations. JAX timings
synchronize returned arrays where relevant. The first call is reported
separately from the median of subsequent calls, including cache reuse and
JAX compilation effects. The materialized-map composition case checks the
cost of applying a completed grid contraction inside a new composition.
