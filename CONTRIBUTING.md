# Contributing

Create an environment with Python 3.10 or newer and install the test extra:

```bash
python -m pip install -e ".[test]"
ruff check .
pytest
```

Optional JAX behavior is tested with `python -m pip install -e ".[test,jax]"`.

Kompe's dependency direction is intentionally one-way: the package must not
import PynaMIT, Lompe, or secsy. Compatibility names and object translation
belong in those consumer packages. New public behavior should include focused
tests and, where it changes an interface, a migration note.
