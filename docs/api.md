# Public API guide

Kompe's top-level namespace contains representation and geometry types. The
backend-neutral numerical layer is intentionally namespaced under
`kompe.math`; low-level SECS kernels are under `kompe.secs`.

## Constant linear evolution

`prepare_linear_evolution(A, b, method="exponential")` prepares the equation
`x' = A x + b` and returns a reusable `advance(initial, times)` iterator.
`A` is a square `LinearMap` (or matrix); `b` and the initial state have its
scientific input shape. Offsets `times` are non-negative and strictly increasing.
The iterator returns arrays with shape `A.input_shape + (n_samples,)`, in
blocks of at most `batch_size` samples (default 32). A batch is a sequence of
output times, not an ensemble of initial conditions.
For example:

```python
from kompe.math import diagonal_linear_map, get_array_module, prepare_linear_evolution

xp = get_array_module()
A = diagonal_linear_map(xp.array([-1.0, 0.0]))
advance = prepare_linear_evolution(A, xp.array([1.0, 2.0]))
for samples in advance(xp.zeros(2), [0.0, 0.1, 0.5], batch_size=2):
    print(samples)  # (2, 2), then (2, 1)
```

Euler (`method="euler", dt=...`) uses fixed steps and a linear within-step
interpolant. SciPy's `RK23`, `RK45`, `DOP853`, `Radau`, `BDF`, and `LSODA`
adapt over the complete requested interval; observations do not restart them.
Their `rtol` and `atol` use state units, and `atol` can have the state shape.
Exponential evolution uses `affine_exponential` and a bounded duration cache,
without solving for an equilibrium. Optional `output_interval` on `advance`
identifies a regular output cadence despite timestamp-subtraction roundoff.

Euler preserves diagonal and matrix-free forms; exponential propagation keeps
diagonals but materializes other maps. SciPy is an explicit CPU boundary:
the fixed matrix, forcing and tolerance are transferred at preparation, and
initial states at each call. Applications return the prepared equation's
backend arrays. Changing `A` or `b` requires another preparation; the caller
owns input-change timestamps and physical unit conversions.

Array-backed Euler and exponential evolution execute output blocks inside
JAX. SciPy keeps one adaptive solver across blocks and transfers complete
sample blocks to the selected backend. Changing batch size never restarts
accepted steps. If later integration fails, completed blocks remain usable;
a sequential solver also yields any completed partial block before raising.

## Representations and meshes

- `SphericalGrid`: immutable spherical sample points in the caller's coordinate
  frame, optionally with area weights. Coordinates are stored flat while
  `shape` retains the broadcast input shape for reshaping evaluated arrays.
- `SHBasis`: real spherical-harmonic scalar and Helmholtz expansion.
- `SECSBasis`: curl-free or divergence-free elementary-current expansion;
  construction requires an explicit `current_type`, and every synthesis
  method follows that mode. Its current operator accepts `chunk_size` for
  bounded-memory forward and adjoint evaluation.
- `GlobalCSBasis`: closed-sphere, cell-centred cubed-sphere expansion. Its
  `cells_per_edge` resolution (or an existing mesh) is required.
- `GlobalCSMesh`: immutable six-face geometry used by `GlobalCSBasis`.
- `GlobalCSOperators`: available as `mesh.operators`; native gradient,
  Helmholtz, Laplacian and mean-free Poisson maps for cell-centre samples.
- `GlobalCSRemapper`: geometry-only interpolation independent of coefficient
  bases. `scalar_operator(source, target)` and `tangential_operator(source, target)`
  return cached sparse maps. Direct `interpolate_vector()` accepts and returns
  `(theta, phi, radial)` components on the CPU, with optional trailing batch axes.
- `GlobalCSProjection`: stateless six-face coordinate and component maps. Its
  physical vector transforms explicitly map ENU components to and from local
  `(xi, eta, radial)` components.
- `RegionalCSProjection`: rotated regional gnomonic coordinate chart.
- `RegionalCSMesh`: structured bounded mesh with an explicit radius.
- `RegionalCSMeshSpec`: versioned serialization boundary for regional meshes.

`RegionalCSMesh(..., shape=(n_eta, n_xi))` follows NumPy array order. When
specifying physical resolution, use the named `xi_cell_size=` and
`eta_cell_size=` keywords; xi is parallel to the projection orientation and eta
is perpendicular to it. `RegionalCSMesh.from_edges(...)` is the exact-geometry
constructor used by serialization and derived meshes.
- `RegionalCSOperators`: available as `mesh.operators`; owns interpolation,
  gradients, and divergence. Its public surface
  vectors use `(theta, phi)`; `coordinate_derivative_matrices()` explicitly
  exposes derivatives with respect to the local `(xi, eta)` chart.
- `SolidHarmonicOperators`: regular/irregular radial continuation for an
  `SHBasis`; these are not surface-basis operations.

SH surface gradients use the analytic limit at each pole, with tangential
components expressed in the direction of the supplied longitude. Nearby points
are evaluated at their actual coordinates without a sine floor or pole shift.
Both `legendre_method="internal"` and `"scipy"` follow the same convention.
The even, cell-centred global cubed-sphere mesh does not sample either pole.

## Analysis and synthesis

`SphericalTransform(basis, grid)` binds a closed-surface basis to evaluation
points. `synthesize_scalar` and `synthesize_helmholtz` map coefficients to
samples. `analyze_scalar` and `analyze_helmholtz` fit samples already on the
bound grid. `analyze_scalar_samples` and `analyze_helmholtz_samples` fit on an
explicit `input_grid`. They return coefficients in the transform's basis,
with the same layout as bound-grid analysis and synthesis.

Remapping is a separate mathematical operation. Supply `remap=operator` to
map samples from `input_grid` to the transform's grid before fitting. A scalar
map has domain `(input_grid.size,)` and codomain `(transform.grid.size,)`;
a tangential map has domain `(2, input_grid.size)` and codomain
`(2, transform.grid.size)`. For example:

```python
from kompe import GlobalCSRemapper

remapper = GlobalCSRemapper()
remap = remapper.tangential_operator(input_grid, transform.grid)
coefficients = transform.analyze_helmholtz_samples(values, input_grid=input_grid, remap=remap)
```

Any compatible `LinearMap` can be supplied; it need not come from a CS remapper.
An explicit map is always applied, even when the grid coordinates match.
The CS remapping factories return identity maps for matching grids and cache
interpolation for different grids. They preserve vector components through
their coordinate changes. Without a map, no remapping or basis-dependent
route is guessed. CS Helmholtz fits off their native grid can use a remap
to a native-grid transform; scalar CS fits can also fit the interpolation
operator directly on observation points.

Sample fits inherit the transform's regularization when `reg_lambda` is omitted
or `None`; pass `0` to disable it. Omitted `sqrt_weights` reuse explicit transform
weights on the same grid. A direct fit on another grid uses weights supplied for
that grid, or its supplied `area_weights` when `area_weighted=True`; bound-grid
weights are never applied to different points just because the array sizes match.
`area_weighted=True` requires `grid.area_weights` unless explicit residual weights
are supplied. For a grid uniform in theta and phi, area weights are proportional
to `sin(theta)`; for equal-area sampling they are constant. Arbitrary coordinates
do not define either sampling measure, so Kompe does not guess one.
Source-grid `sqrt_weights` cannot be carried through a grid remap; configure
target-grid weights on the transform instead. `with_basis` returns a transform
for another coefficient basis while reusing the same grid and numerical policy.
An explicitly different evaluation algorithm is retained even when its
coefficient layout is compatible with the previous basis.
Sample analysis reuses the bound transform's operators and factorizations when
the basis, exact grid and measure, weights, and regularization match. Only
different fit settings need another cached transform. Reusing the bound fit
requires no extra hashing or host-device transfer of its inherited weights.

All analysis methods use leading scientific axes and trailing batch axes:
scalar data `(n_points, *batch_shape)` produce `(n_coeffs, *batch_shape)`;
tangential data `(2, n_points, *batch_shape)` produce
`(2, n_coeffs, *batch_shape)`. An unbatched input stays unbatched. A flat
single-field vector is also accepted, as in ordinary linear algebra.
Identity shortcuts, optimized operators, explicit least-squares solvers,
and remapping follow this same convention. Flatten shaped spatial grids
explicitly, and use `xp.moveaxis` at a provider/storage boundary for time-first
data. Kompe does not infer axis meanings from coincident dimension lengths.

`gradient_theta_array`/`gradient_theta_operator` evaluate `d/dtheta`.
`gradient_phi_array`/`gradient_phi_operator` evaluate `(1/sin(theta)) d/dphi`,
not the bare azimuthal partial derivative. These are unit-sphere gradient
components, with angles differentiated in radians. Divide by physical radius
for spatial derivatives. The corresponding scalar-evaluation and synthesis
calls accept `gradient_component="theta"` or `"phi"`; omitted means scalar
values. `surface_gradient_operator` returns both components in `(theta, phi)`
order.

Basis evaluation is operator-first: implement
`scalar_evaluation_operator(grid, gradient_component=None, *, persist=True)`
when defining a new `ScalarBasis`. The default `scalar_evaluation_array`
materializes that map only on request. Existing analytical SH assembly stays
direct; CS evaluation and coefficient subsets retain their sparse actions.
Gradients map coefficients directly to sampled vector components. They need
not lie in the original scalar expansion and do not require a coefficient fit.

Scalar synthesis and fitting accept any `ScalarBasis`, including `SECSBasis`.
Only Helmholtz operations and surface-smoothness regularization require
`SurfaceDifferentialBasis`; these fail explicitly when unsupported. Gradient
evaluation is available when the scalar basis provides the derivative components.

Basis scalar and vector evaluation calls accept `persist=False` to forbid
evaluation disk-cache reads and writes. On a transform, set
`use_persistent_evaluation_cache=False` for the same policy across all evaluations.
Bounded in-memory reuse remains enabled; no disk access is needed to materialize
an already constructed map. SH operators share derivative components until an
explicit stacked/rotated array is requested. Requested arrays are cached by the
map and used on subsequent applications.

`surface_laplacian_operator(r)` instead returns coefficients in the **same**
space. SH modes are closed under this operation, with diagonal eigenvalues
`-n*(n+1)/r**2`. An arbitrary CS coefficient subset is not closed: its Laplacian
can be nonzero at omitted nodes. Use `laplacian_evaluation_operator(grid, r)`
for the complete sampled derivative, or explicitly compose a projection if
an approximation in a smaller space is intended. Such subsets reject the
same-space operation instead of silently truncating its output.

`scalar_smoothness_operator()` and `helmholtz_smoothness_operator()` return
`LinearMap` penalties. Their squared residual norms are unit-sphere surface
means of `|grad(f)|**2` and `|div(v)|**2 + |curl_r(v)|**2`, respectively. The
latter is a div–curl energy, not the covariant vector-gradient energy. SH
penalties are diagonal modal weights; CS penalties are area-weighted sparse
gradient and Laplacian maps. The residual axes can therefore differ between
bases. Subsets restrict coefficient inputs, not the spatial region where
smoothness is measured. `SphericalTransform(reg_lambda=...)` uses these maps
with its existing relative normalization; solver choice is unchanged.

Explicit representations that retain scientific axes end in `_array`, even
when a scalar case happens to be 2-D. Flat linear-algebra representations that
are always exactly 2-D end in `_matrix`; structured equivalents end in
`_operator`. For example, `scalar_synthesis_array`,
`helmholtz_synthesis_array`, and `helmholtz_synthesis_operator` state both their
representation and semantic direction. Vectorized geometric component
transforms likewise end in `_array` because sample axes change their rank.
They spell out both directions instead of changing meaning through an
`inverse` flag.

The word *projection* in Kompe's type vocabulary refers to coordinate charts.
Mean-free coefficient projections retain that mathematical name because they
are explicit projections within one coefficient space.

For a surface basis, `basis.mean_free` means every represented scalar field
has zero surface mean. In contrast, `basis.omits_constant_mode()` means the
space cannot represent a nonzero constant: a restricted CS nodal basis can
omit the constant while still representing fields with nonzero means.
`CoefficientSpace.mean_free` is a normalization policy, defaulting to the
basis's intrinsic `mean_free` status. `basis.scalar_mean(coeffs)` evaluates
the physical mean with coefficients on the first axis, followed by any batch
axes. Mean-free projection subtracts that mean times
`scalar_constant_coefficients`, or does nothing
for an intrinsically mean-free basis. If a non-mean-free subset cannot
represent this constant shift, impose a mean constraint during fitting;
attempting to subtract it within that subset raises an error.

`CoefficientSpace.project_mean_free` and `validate_coefficients` accept
`space.shape + batch_shape`, or `(space.size,) + batch_shape` with flattened
field axes. `project_helmholtz_mean_free` likewise accepts
`(2, n_coefficients, *batch_shape)` or `(2*n_coefficients, *batch_shape)`.
Analysis, normalization and synthesis compose without moving axes.
Values are ordinary NumPy/JAX arrays, not a separate coefficient container.
The receiving basis defines their meaning; use coefficient-space signatures
when exchanging arrays between independently constructed bases. Ownership and
copying belong where values become persistent state. Time-first storage belongs
at the consumer's boundary, not inside coefficient normalization.

## Numerical layer

`kompe.math.LinearMap` is the common structured-operator contract. Use
`as_linear_map` to wrap dense or sparse arrays and the named constructors for
diagonal, identity, pointwise, stacked, or indexed maps. Least-squares helpers
consume `LinearMap` without requiring dense materialization.
Call `operator(values)` for shaped application: values have
`operator.input_shape + batch_shape`, and the result has
`operator.output_shape + batch_shape`. Domain axes come first; arbitrary
trailing batch axes are retained and use one block application. `@`, `matvec`,
and `matmat` keep their flat linear-algebra semantics. For example, a Helmholtz
synthesis map accepts `(2, n_coefficients, n_times)` and returns
`(2, n_points, n_times)`. Maps have identity-based equality and hashing, like
functions, and can be passed directly to `jax.jit`.
`matmat` and `rmatmat` accept a 2-D column block or a single 1-D vector,
returning the same rank regardless of materialization. Empty column blocks
and empty domains or codomains need no action callbacks.
Custom matrix-free maps use the ordinary constructor keywords `matvec`,
`rmatvec`, and optional `matmat`, `rmatmat`, `dense_array`, or `diagonal`
functions; implementation storage remains private.
`to_matrix()` returns the flat 2-D representation; `to_array()` returns the
same values with shaped domain and codomain axes. Both cache the dense
representation instead of repeating its construction during eager use.
Materializations and dense least-squares factors produced inside `jax.jit`
remain in the compiled calculation, not the reusable Python caches; later
interactive calls can still build and retain concrete values. Materialize
or prepare a fixed operator outside JIT when its construction should be
shared across separate compiled functions. SciPy sparse maps transfer and
retain their static sparse structure even when first applied under JIT.
New compositions and scalar multiples reuse contractions already materialized on the active
backend instead of expanding their original tensor factors again. Existing
compositions retain the representation with which they were constructed.
Known diagonal and identity maps remain vector-backed even when a full matrix
has been requested for inspection.
`materialized_matrix` exposes an existing dense representation, or `None`,
without constructing values or transferring arrays. It prefers the active
backend's cached copy, then any existing dense values. Numerical kernels can
therefore reuse explicit arrays without forcing matrix-free maps to become
dense.
`diagonal()` returns the scale vector only when a diagonal representation is
known. It does not materialize a matrix or transfer it to the CPU to discover
structure. Declare diagonal maps with `diagonal_linear_map(values)`; use
`xp.diag(map.to_matrix())` for the diagonal of a general dense matrix.

`A.normal_operator(row_scale=w)` returns the self-adjoint map
`A* diag(abs(w)**2) A`; omitting weights gives `A* A`. It maps A's
`input_shape` to itself and accepts trailing batch axes. Application stays
matrix-free, diagonal maps stay vector-backed, and sparse maps retain an
explicit sparse representation. Materialization reuses an existing dense A,
then a declared normal product, then sparse structure, or bounded column
blocks without a rectangular matrix. Materializing the returned map caches
that normal matrix for subsequent application.

A custom `normal_matrix(xp, row_scale)` constructor callback supplies the
flat square product on the requested backend. It must describe the same
fixed action as the operator, including row weights. Row/column scaling,
vertical stacking, and coordinate restrictions preserve these products;
in particular, `(A @ Z)* (A @ Z) = Z* (A* A) Z`. Arbitrary sums cannot
sum normal products because they also contain cross terms.

`normal_matrix_diag()` computes `diag(A* A)` for scaling and preconditioning.
Optional `row_scale=w` computes `diag(A* diag(abs(w)**2) A)`, where `w` has
one value per flat output row. Diagonal vectors remain compact; otherwise an
existing dense representation is reduced on its backend, transferring only
the resulting vector to the CPU. Structured formulas handle sparse maps,
component maps, selections, and their row/column scalings without coefficient
probes. A custom `normal_matrix_diag` constructor callback accepts the same
optional `row_scale=None` keyword. Without either representation, the map probes
the operator in bounded column blocks, without requesting a dense matrix.
Tensor contractions and probed sample blocks are reduced on their numerical
backend before transferring the resulting diagonal to the CPU.

`LeastSquaresProblem(A)` takes its solution and data shapes from `A`'s maps;
several data maps must share one `input_shape`, but may have different
`output_shape`s. Raw matrices retain their ordinary flat row/column axes.
Label scientific axes once with
`as_linear_map(array, input_shape=..., output_shape=...)` before constructing
the problem. There are no separate shape arguments on `LeastSquaresProblem`.
No operator is evaluated or materialized to determine these shapes.

For exact constraints `C x = 0`, supply the constraint rows, not penalty rows:

```python
problem = LeastSquaresProblem(A, sqrt_weights=weights, regularization=R, constraints=C)
x = solver.solve(problem, values)
```

`C` may be a dense array or a SciPy sparse matrix. Sparse `normal_solve`
retains `C` in a sparse KKT system. Other methods use
`problem.solution_basis`, an orthonormal null-space `LinearMap` retained as
compact QR reflectors, followed by backend-native application. The reduced
objective is inspectable as `problem.reduced_problem`; coefficient norms and
spectral cutoffs retain their usual meaning. `restrict_solution(Z)` also remains
available for explicit coordinate changes. SH and CS Helmholtz fits declare
their zero-mean constraints on `helmholtz_least_squares_problem`. A subset is
gauged only when it retains
the full support of `basis.scalar_constant_coefficients`; having nonzero mean
weights alone does not imply an unobservable constant.

Constraint row units are normalized for numerical setup without changing
`C x = 0`. Restricting coordinates carries the remaining row space of `C Z`;
fully or partly eliminated rows are removed at the product's roundoff scale,
not the least-squares tolerance. This works with independently constructed
null spaces, without relying on Python object identity. Compact null-space
maps provide their normal diagonal analytically for this setup.

Choose an algorithm explicitly with `LeastSquaresSolver(method="svd", tolerance=...)`
and reuse it in `transform.analyze_scalar(values, solver=solver)` or any of the
Helmholtz/sample-analysis methods. Omitted solvers use
`KOMPE_LEAST_SQUARES_SOLVER`, defaulting to `normal_pinv`. A reusable solver
created with `LeastSquaresSolver()` reads that same default at construction;
later environment changes do not change its `.method`. A reusable solver
supplies its own tolerance; otherwise the transform's `tolerance` is used.
Configured iterative preconditioners apply in both `solve` and `prepare` and
are cached per problem. Passing an explicit map overrides the configured kind.
Direct and pseudoinverse methods reject preconditioners rather than ignoring them.
Native unweighted, unregularized scalar nodal samples already are coefficients
and require no solve.

The default tolerance is `1e-15`, with algorithm-specific meaning:

- `svd`: relative cutoff on singular values of the weighted, augmented system.
- `normal_pinv`: relative cutoff on normal-matrix eigenvalues, i.e. **squared**
  singular values. Its historical meaning is unchanged.
- `lsmr`: relative residual and normal-residual convergence tolerances.
- `cgls`: relative convergence tolerance for the normal equations.
- `normal_solve`: direct normal-equation solve, without spectral truncation;
  tolerance does not affect the result. The normal matrix must be nonsingular
  after removing known gauges. Unregularized Helmholtz fits reuse an available
  sparse constrained LU or full-rank Cholesky factorization of that objective.
  Sparse regularized problems retain a sparse normal/KKT factorization too;
  diagonal problems use vector division instead of matrix factorization.

Tolerance does not switch algorithms. Choose iterative methods for large
matrix-free fits, SVD when avoiding normal-equation conditioning is important,
and normal methods when their cost/conditioning tradeoff suits the problem.
These choices are not inferred from the data or silently substituted.
LSMR rescales the complete objective using its initial adjoint norm, including
any damping, so a change of units does not trigger a spurious condition-limit
stop. This scaling acts on operator results and does not copy the dense data
matrix for each right-hand side. Sparse constrained maps likewise remove a
common residual scale before forming their KKT system. Neither operation
changes relative weights, the minimizer, or the selected tolerance.

`helmholtz_analysis_operator` is a different contract: a fixed geometric
inverse suitable for repeated composition and evolution operators. It retains
a Cholesky factor for full-rank SH or a sparse constrained LU factor for native
CS. These are CPU construction boundaries, and the CS sparse solves remain on
the CPU even with JAX inputs. Rank-deficient fixed SH inverses use a
pseudoinverse. Its constrained coordinates come from the same least-squares
problem as configurable analysis, including non-native CS and coefficient
subsets. Both enforce the physical surface-mean gauge, not a different
Euclidean coefficient mean. Changing the fit solver does not change these fixed maps.

SH Helmholtz arrays, basis operators, and transform operators share one dense
materialization. Until an array is requested, the operator retains its
structured action. Dense synthesis assembles the Helmholtz blocks directly,
without multiplying an identity matrix.

The problem exposes `data_operators` and `weight_operators` as lists
of maps, separate from the assembled weighted `data_operator` and regularized
`system_operator`. Relative regularization requires a nonzero weighted data
operator and an explicit nonzero regularizer; it never invents a unit scale or
silently discards a positive strength.

`problem.svd(backend=None)` returns the reduced `U, s, Vh` factors and caches
them separately for NumPy and JAX. SVD solves and spectral preconditioners
reuse these same factors; JAX solves do not round-trip through NumPy.

`normal_solve` likewise retains its LU factors for repeated right-hand sides,
with separate entries for the execution backend and RHS precision. The
algorithm remains a pivoted-LU direct solve. Normal solves
do not append zero RHS rows for regularization: the penalty enters the normal
matrix, while only the data contribute to the right-hand side. Once the factors
are available, the temporary normal and augmented matrices are released; the
data operator remains available for repeated adjoint products.

Declared sparse structure survives `LinearMap` weighting, scaling, stacking,
selection, adjoints, and composition. `map.is_sparse` inspects that capability;
`map.to_sparse_matrix()` explicitly materializes a SciPy CSR representation,
never a dense matrix or column probes. Sparse direct factorization and solves
are CPU operations, including when JAX calls them through a host callback.
The callback handles an entire RHS block. This is not a GPU sparse direct solver.

Fit configuration is fixed: create a new problem or transform to change its
operators or weights. Array-valued fit weights are owned (NumPy read-only,
JAX immutable), including when converting a mutable NumPy buffer to JAX.
User-supplied `LinearMap` actions must remain fixed while their fit is in use;
Kompe does not copy or fingerprint a large matrix on every solve.

The normal-equation helpers `dense_normal_matrix` and `dense_normal_pinv`
also accept `backend=`. Dense solvers infer the
execution backend from all active operators and the right-hand side, even
when only the regularizer is a JAX array. Prepared `normal_pinv` response solvers
retain their factorization and reuse backend-specific materializations when
later right-hand sides select another backend. Their data
adjoint retains its sparse or matrix-free action: preparation does not force
a dense rectangular data map. Existing materializations are still reused.
Normal products belong to operators: `LeastSquaresProblem` constructs
`A* W* W A + sum(L* L)` through the regularized system's `normal_operator()`.
There is no second description of the data operator on the problem.
Diagonal penalties remain vectors, and coordinate restrictions apply on the
array backend without expanding a compact constraint basis. Iterative fits
do not construct these products. An explicit NumPy request is a CPU boundary.
The stack promotes to its combined dtype before adding fractional penalties;
single-precision inputs remain single precision unless another operand widens them.

SH scalar and Helmholtz fits use memory-bounded structured construction on
both NumPy and JAX. Equal component weights share products. Eager JAX setup transfers one
equality boolean to recognize this structure, not weight or field arrays.
Under JIT, traced weights use the full expression without host synchronization.
Relative regularization is constructed explicitly from normal diagonals,
without constructing a normal matrix. Declaring a normal product
does not make iterative fitting construct a full normal matrix. Fixed SH
Cholesky analysis still crosses an explicit CPU factorization boundary after
normal construction.

JAX LSMR and CGLS use compiled device-side iterations. Batching keeps each RHS's
convergence independent; it is not one block/Frobenius stopping criterion.
Compiled kernels are owned and reused by the problem, so discarded problems
do not remain alive in a global static-operator cache. LSMR reports convergence
diagnostics only after solving, not through per-iteration CPU synchronization.
NumPy continues to use SciPy. JAX iterative maps must have JIT-compatible actions.
JAX CGLS checks the true normal residual after the solve, since JAX CG does not
return a convergence status. It warns when the requested tolerance is unmet,
including tolerances below the achieved floating-point accuracy. There is no
per-iteration host synchronization.

Iterative `x0` values are always in the original coefficient space, even with
constraints or a preconditioner. Supply one field (or flat vector) to share
across the RHSs, or `solution_shape + rhs_batch_shape` for individual initial
guesses. LSMR solves `A P y = b - A x0` and recovers `x = x0 + P y` when a
right preconditioner is present. An explicit penalty R participates in that
correction equation and remains centred on zero. Constrained preconditioners
and their compiled kernels are reused by both `solve()` and `prepare()`.

`LeastSquaresSolver.solve(..., **options)` accepts options supported by its
selected iterative algorithm. Dense algorithms (`svd`, `normal_solve`, and
`normal_pinv`) reject extra options instead of silently ignoring them.
LSMR supports `damp=sqrt(reg_lambda)` for coefficient-norm regularization
without augmented rows when `x0` is omitted. With a nonzero initial guess,
SciPy's LSMR recurrence penalizes the correction `x - x0`; put an absolute
zero-centred penalty in the problem to retain it while using an initial guess.
Nonzero damping cannot be combined with a right
preconditioner: that would penalize the transformed coordinates instead.
Represent the penalty explicitly in `LeastSquaresProblem` for such solves.
`LeastSquaresProblem(A, regularization=R)` uses exactly `||W(Ax-b)||² + ||Rx||²`.
`regularization` may also be a list of already-scaled operators. An absolute
penalty `lambda * ||L x||²` is therefore `regularization=sqrt(lambda) * L`.
For the relative convention, use
`R = relative_regularization(A, L, strength, sqrt_weights=weights)` and pass
the same data operator/weights to the problem. This explicit setup step
balances median positive normal diagonals. `SphericalTransform.reg_lambda`
continues to use that relative convention. Solver methods and tolerances are
unchanged; exact coordinate restrictions compose R with the constraint basis.

NumPy/SciPy is the reference backend. JAX is optional and lazy. Select it with
`kompe.math.set_backend("jax")`, a `backend_context("jax")`, or explicit JAX
arrays. Kompe never changes JAX's global 64-bit setting.
`get_array_module(*arrays, backend=None)` uses an explicit backend when given,
otherwise JAX operands take precedence over the configured default. For
example, `get_array_module(values, backend="numpy")` selects NumPy without
changing the global setting; converting the values remains explicit.
`get_backend(*arrays)` reports the corresponding `"numpy"` or `"jax"` name;
with no operands, `get_backend()` reports the configured default.

## Sampled derivatives

`kompe.math.centered_derivative(coordinates, values, half_window_points=1)`
differentiates the last values axis on finite, strictly increasing coordinates.
It preserves arbitrary leading batch axes and the NumPy/JAX backend, including
JAX tracing and differentiation. At each interior sample it differentiates the
quadratic through that sample and its two neighbours, the requested number of
indices away. It is therefore quadratic-exact even on uneven coordinates;
incomplete or missing stencils remain NaN. Coordinate units determine the
derivative units. Datetime decoding and resampling are application concerns,
not part of this numerical primitive.

## Cache lifecycle

`SHBasis`, `GlobalCSBasis`, and `SphericalTransform` expose `cache_info()` and
`clear_cache()`. Caches use exact representation/grid signatures; shared CS
remap matrices and per-basis target-grid caches are bounded. Long-running
applications can clear them at known lifecycle boundaries without changing
numerical results.

`SphericalTransform.cache_info()` reports `cached_attributes`,
`scalar_problem_cached`, and `helmholtz_problem_cached`. These describe
cached Python values and problem definitions, not dense materializations or
completed factorizations.

Each concrete basis defines `coefficient_space_signature` from its mathematical
layout and normalization. `signature` additionally identifies evaluation
details such as the SH Legendre algorithm. The base class does not inspect
subclass-specific parameter names to guess identity. `root_basis` returns the
underlying basis before any nested coefficient subsets.
