# Kompe microbenchmarks

Run `python benchmarks/benchmark_core.py` from an installed checkout. The
script reports construction and operator-materialization timings for small,
representative global CS, regional CS, SH, and SECS problems. It is intended
for before/after comparisons, not as a hardware-independent pass/fail test.

Use the same Python environment, backend, machine load, and command arguments
for both revisions. JAX timings synchronize returned arrays where relevant;
the first iteration includes compilation and subsequent iterations measure
warm execution.
