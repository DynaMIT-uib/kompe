# Kompe development style

Write mathematical and numerical code for scientists who read, test, and compose it
interactively.

- Put mathematical objects, equations, units, and coordinate conventions before software
  machinery. Build reusable primitives for basis evaluation, coordinate transformations,
  linear operators, interpolation, solvers, and backend-neutral array operations. Define each
  at the mathematical level where its contract is complete, then let higher-level code compose
  those primitives so the mathematical action remains visible.
- Keep Kompe independent of PynaMIT's physical model. Functionality belongs here when it is
  useful and meaningful as general spherical or numerical mathematics without MIT-specific
  concepts; reuse alone is not a reason to widen Kompe's scope.
- Organize modules around cohesive mathematical, numerical, or infrastructure roles, not around
  line count or one class per file. Split a module when it mixes independently understandable
  calculations or state ownership, but prefer shallow directories and direct navigation. Avoid
  forwarding-only modules, generic utility collections, and fragmented call chains. Keep
  caching, serialization, compatibility, and backend handling at clearly named boundaries.
- Use `theta` (colatitude) and `phi` (longitude) for internal spherical coordinates. Canonical
  tangential components are ordered `(theta, phi)`, equivalent to `(south, east)`. Use
  latitude/longitude and east/north at geographic interfaces, state angle units explicitly,
  and make conversions visible.
- Keep mathematical roles precise: a grid stores evaluation points, a mesh owns topology and
  geometry, a basis maps coefficients to fields and derivatives, a projection maps coordinates,
  and a transform performs analysis and synthesis. Do not give one object several unrelated
  roles merely for convenience.
- Prefer direct equations and ordinary Python statements. Keep a helper only when it is reused,
  isolates a difficult backend or numerical boundary, or makes an equation clearer. Delete
  forwarding-only and single-use helpers when inlining is easier to follow.
- Use a class or other abstraction when it represents a stable mathematical concept, owns
  meaningful reusable state, isolates a necessary boundary, or removes concrete duplication.
  Do not add a dataclass, protocol, registry, wrapper, or object merely to relay calls that a
  function, dictionary, array, or direct expression states plainly.
- Validate public inputs, coordinate and unit conversions, and numerical assumptions once.
  Internal routines should trust normalized values instead of accumulating defensive checks
  and fallback paths.
- Make unsupported behavior fail explicitly. Do not guess a basis, shape, coordinate system,
  or algorithm when the caller has not provided enough information.
- Use one canonical public name and representation for each mathematical operation. Retain
  standard symbols when they make comparison with equations easier, and prefer plain,
  descriptive names for software-only concepts. Keep any necessary compatibility alias at a
  public boundary rather than propagating it through the implementation.
- Name transformations by their mathematical direction. Use projection or analysis for sampled
  values to coefficients, and evaluation or synthesis for coefficients to field values. Make
  coordinate systems, component order, domains, and codomains explicit at transformation
  boundaries.
- Use Kompe's configured array-backend interface for operations supported by NumPy and JAX.
  Do not add isolated NumPy/JAX branches or silently transfer device arrays to NumPy. Keep
  SciPy algorithms, serialization, and plotting at explicit CPU boundaries, and make necessary
  conversions visible.
- Preserve useful operator structure such as diagonal, sparse, or matrix-free forms. Materialize
  an operator only when repeated use benefits from it, reuse that materialization afterward,
  and do not turn a diagonal vector into a dense matrix without a numerical reason.
- Keep arrays, operators, bases, grids, meshes, projections, and transforms readily inspectable
  from IPython. Composition should expose the mathematical action rather than hide it behind
  dispatch or configuration machinery.
- Test mathematical identities, coordinate conventions, operator action, numerical accuracy,
  and rank-deficient or boundary cases at the level where they matter. Exercise portable
  numerical paths with both NumPy and JAX, and test structured operators without requiring
  unnecessary dense materialization.

Preserve numerical behavior, performance, and backend compatibility, and run the relevant tests
after changes.
