# Kompe development style

Write numerical code for scientists who read, test, and compose it interactively.

- Prefer direct equations and ordinary Python statements. Keep a helper only when it is reused,
  isolates a difficult backend or numerical boundary, or makes an equation clearer.
- Do not add a class, dataclass, protocol, registry, or wrapper when a function, dictionary,
  array, or direct call says the same thing plainly.
- Validate public inputs and numerical assumptions once. Internal routines should trust those
  normalized values instead of accumulating repeated defensive checks and fallback paths.
- Make unsupported behavior fail explicitly. Do not guess a backend, basis, shape, or algorithm
  when the caller has not provided enough information.
- Keep arrays, operators, bases, grids, and transforms inspectable from IPython. Abstractions
  should correspond to real mathematical roles or ownership of reusable state.
- Retain standard scientific symbols when they make comparison with equations easier. Prefer
  plain, descriptive names for software-only concepts.
- Add an abstraction only after identifying the concrete duplication or state ownership it
  removes. Delete forwarding-only and single-use helpers when inlining is clearer.

Preserve numerical behavior and backend compatibility, and run the relevant tests after changes.
