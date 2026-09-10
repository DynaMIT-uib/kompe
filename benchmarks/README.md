# Kompe microbenchmarks

## Conservative global-CS investigation

`benchmark_conservative_cs.py` constructs a weighted-adjoint divergence from
the existing gradient G and cell areas M, then L = -M^-1 G* diag(M, M) G.
This is a research prototype, not a selectable production discretization.
It checks integration by parts and conservation independently of harmonic
accuracy, then reports sparsity, synchronized application timings, and a
small-grid spectrum. Common gradient construction is timed separately.

```sh
PYTHONPATH=src python benchmarks/benchmark_conservative_cs.py --backend numpy
PYTHONPATH=src JAX_ENABLE_X64=1 python benchmarks/benchmark_conservative_cs.py --backend jax
```

On 9 September 2026, the prototype conserved to roundoff but failed the
pointwise-convergence test. Area-weighted relative RMS error for the l=1
harmonic was:

| Cells per edge | Current collocated L | Weighted-adjoint L |
| --- | --- | --- |
| 8 | 0.0276 | 0.5846 |
| 16 | 0.00836 | 0.5691 |
| 32 | 0.00267 | 0.5606 |

At resolution 32 the non-axisymmetric l=3, m=2 error was 0.00606 versus
0.3148. Exact conservation alone is therefore insufficient: simply using
the weighted adjoint of these collocated stencils is not a drop-in accurate
divergence. The global gradient and face-interface treatment need a compatible
redesign. Curl identities have not been established by this prototype either.
Production keeps its existing convergent discretization. See
[Thuburn, Cotter and Dubos (2014)](https://gmd.copernicus.org/articles/7/909/2014/)
for a related mimetic finite-volume direction and its accuracy tradeoffs.

## Core construction

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

## Weighted normal-matrix construction and fitting

`benchmark_helmholtz_gram.py` compares constructing a weighted normal matrix
from the full rectangular Helmholtz synthesis matrix with constructing its
blocks directly from the two scalar derivative arrays using the production
builder. It checks numerical equivalence before timing. The arrays are
synthetic: this isolates contractions, not basis evaluation, gauge constraints,
or a complete fit. There is no separate prototype implementation to maintain.

```sh
JAX_ENABLE_X64=1 python benchmarks/benchmark_helmholtz_gram.py --backend jax
```

At 1,728 points and 440 scalar coefficients on JAX CPU, one local warm run
measured 14.0 ms for the explicit calculation and 9.3 ms for structured
assembly. Compiler-reported temporary workspace fell from 48.7 MB to
22.9 MB. These are not process peak-memory measurements: arguments, outputs,
and compiled executables are excluded. No GPU speedup is established.

`benchmark_normal_fit.py` measures complete SH fits with actual derivative
arrays, unequal component weights, relative smoothness regularization of
0.001, and exact mean gauges. It checks solutions against an independent
augmented NumPy least-squares calculation. It reports the first fit, fresh
fits after compilation warm-up, and repeated RHS calls reusing a fit.
Basis evaluation is shared and excluded; no disk cache is involved.

```sh
python benchmarks/benchmark_normal_fit.py --backend numpy
JAX_ENABLE_X64=1 python benchmarks/benchmark_normal_fit.py --backend jax
```

On 8 September 2026, macOS arm64 CPU, Python 3.12.13 and JAX 0.9.2, a local
before/after comparison at 1,728 points and 441 scalar coefficients gave:

| JAX CPU method | Fresh fit, warm kernels (ms) | Cached RHS (ms) |
| --- | --- | --- |
| `normal_solve` | 100.5 → 77.2 | 3.8 → 5.3 |
| `normal_pinv` | 217.3 → 233.4 | 2.7 → 3.8 |

These are medians of three before and five after calls, not performance
guarantees. The first `normal_solve` fit including new JAX compilation cost
about 1.9 s after the change, versus 1.2 s before. The retained rectangular
data matrix fell from 24.3 MB to zero in these unprepared fits. This trades
some repeated-adjoint throughput and first-use latency for memory and faster
direct-solve setup; it is not a universal speedup. The corresponding new
NumPy warm fit/RHS times were 103.3/2.2 ms for `normal_solve` and 206.5/2.0 ms
for `normal_pinv`. Solution errors relative to augmented least squares were
about 1e-14 on both backends.

For repeated RHS throughput, the existing `problem.data_operator.to_matrix()`
can retain the weighted data matrix explicitly. Prepared `normal_pinv`
responses retain the structured data adjoint, or reuse its existing materialization. Neither needs another
solver option. Production solver choices, tolerance meanings, and regularization
scaling remain unchanged; iterative fitting does not invoke a normal builder.

The construction savings should not be extrapolated to warm time evolution.
In an immediate PynaMIT Euler comparison (35 poloidal coefficients, 1,000
steps, output every ten), JAX CPU took 79.6 ms with the structured builder
and 79.0 ms with that builder disabled. The warm profile contained no normal
construction: output handling and operator application dominated this case.

The explicit-regularization refactor was checked separately on 9 September
2026 at degree 16, 1,728 points, JAX CPU, five warm repetitions. Before/after
fresh `normal_solve` fits were 42.23/42.20 ms; cached RHS calls were
3.29/3.30 ms. `normal_pinv` fits were 97.90/91.07 ms and cached RHS calls
3.01/2.99 ms. Both versions retained zero bytes of explicit data matrix and
agreed with the independent augmented solution to about 1e-14. This supports
the API simplification without demonstrating a general speedup or GPU result.

## Sparse and compiled iterative fitting

Run the same inputs in separate processes for each revision:

```bash
KOMPE_USE_JAX=1 JAX_ENABLE_X64=1 python benchmarks/benchmark_structured_solvers.py lsmr
KOMPE_USE_JAX=0 python benchmarks/benchmark_structured_solvers.py cs --resolution 16
```

The first case fits eight independent right-hand sides with a 100-by-60
operator. The second fits four regularized native-grid CS Helmholtz fields.
The script reports cold and warm times, explicit dense materializations, and
the solution norm. Mathematical equivalence is covered by the solver tests.

On the development machine's JAX CPU backend (9 September 2026), compiled
LSMR reduced the warm median from about 99 ms to 1.45 ms. The CS sparse path
removed a 6144-by-3070 dense augmented matrix (about 151 MB) and reduced warm
time from 13.2 ms to 5.1 ms; cold fitting was about 0.45 s versus 0.39 s.
These are illustrative measurements, not portable performance thresholds.

## SH derivative reuse

```bash
VECLIB_MAXIMUM_THREADS=1 python benchmarks/benchmark_sh_synthesis.py --backend numpy
JAX_ENABLE_X64=1 python benchmarks/benchmark_sh_synthesis.py --backend jax
```

This compares production synthesis/adjoints to pre-stacked gradient and rotated
gradient arrays, with single fields and batches of 16. At degree 20 and 1,728
points, retained derivative storage is 12.2 MB rather than 36.5 MB. Explicit
materialization is separate and remains available when throughput benefits.

Local NumPy CPU medians (31 repeats, 9 September 2026) were 0.50/0.33 ms for
stacked/shared single-field synthesis and 0.94/0.33 ms for adjoints; batches of
16 were 0.96/0.83 ms and 1.08/0.96 ms. Compiled JAX CPU single-field calls also
improved, while batch timings were close and sometimes slower with shared
derivatives. This is a storage reduction, not a universal speedup or GPU claim.

A separate PynaMIT JAX CPU check (degree 5, Ncs=12, 1,000 Euler steps with
output every ten, 15 warm continuations) measured 72.1 ms before and 75.7 ms
after, including xarray output. Setup was 0.91/0.87 s and first evolution
2.10/2.12 s. Final Br norms agreed to floating-point roundoff. That timing
difference did not reproduce: a repeat measured 72.9/71.3 ms, with identical
warm profile call counts. SH derivative construction was absent from the
warm loop, which already reused prepared response matrices. These results
do not establish a warm-evolution slowdown from derivative sharing.

The profile instead identified per-sample response evaluation and xarray
output handling as substantial costs. PynaMIT's evolution benchmark now
also covers batching these evaluations at the existing write boundary;
that is a separate optimization from SH derivative storage.
