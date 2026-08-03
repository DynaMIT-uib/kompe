# Contributing

Create an environment with Python 3.10 or newer and install the test extra:

```bash
python -m pip install -e ".[test]"
ruff check .
ruff format --check .
pytest
```

Optional JAX behavior is tested with:

```bash
python -m pip install -e ".[test,jax]"
KOMPE_USE_JAX=1 JAX_ENABLE_X64=1 pytest --cov=kompe --cov-fail-under=75
```

Kompe never changes JAX's process-global precision configuration. Applications
that need 64-bit numerical equivalence must enable it before importing JAX or
set `JAX_ENABLE_X64=1`, as the CI command above does.

Kompe's dependency direction is intentionally one-way: the package must not
import PynaMIT, Lompe, or secsy. Compatibility names and object translation
belong in those consumer packages. New public behavior should include focused
tests and, where it changes an interface, a migration note.

Run `python benchmarks/benchmark_core.py` before and after changes to dense
matrix construction, remapping, differentiation, or SECS kernels. Releases are
made only from a `v<version>` tag that matches `pyproject.toml`; the release
workflow reruns all NumPy, JAX, lint, formatting, and package checks before
publishing.
