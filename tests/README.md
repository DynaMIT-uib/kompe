# Test suite

Test files follow Kompe's mathematical objects: coordinate projections,
meshes, bases, transforms, SECS kernels, linear maps, and least-squares
solvers. Large modules keep closely coupled identities together and use short
section headings to separate independent numerical contracts.

`test_basis_operators.py` covers operator-only expansions and coefficient
restrictions without dense parents. `test_smoothness.py` checks spherical
energies against analytical fields and regularized fits against independently
assembled objectives.
`test_constrained_least_squares.py` checks sparse and dense constrained fits
against independent coordinates, including cache ownership and adjoints.
`test_jax_iterative.py` covers compiled LSMR/CGLS, independent RHS convergence,
damping, precision, and compiled-cache lifetime.

Tests should state deterministic coordinates, coefficients, and tolerances
near the equations they exercise. Diagnostic plotting and exploratory random
sampling belong in examples or notebooks unless the plotting operation itself
is under test.

JAX-specific tests use `@pytest.mark.requires_jax`. The shared collection hook
skips that marker when JAX is unavailable, so individual tests do not repeat
backend-detection policy.

Common checks are:

```bash
pytest -q
pytest -q -m requires_jax
ruff check .
ruff format --check .
```
