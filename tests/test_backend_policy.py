"""Tests for optional-backend import and configuration policy."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from kompe.math import backend_context, get_array_module, get_backend, readonly_numpy_array


@pytest.mark.requires_jax
@pytest.mark.parametrize("configured", ["numpy", "jax"])
def test_backend_name_follows_array_module_policy(configured):
    """Backend names and array modules resolve the same operands."""
    import jax.numpy as jnp

    numpy_values = np.ones(1)
    jax_values = jnp.ones(1)
    with backend_context(configured):
        for operands, expected in (
            ((), configured),
            ((numpy_values,), configured),
            ((jax_values,), "jax"),
            ((numpy_values, jax_values), "jax"),
            ((jax_values, numpy_values), "jax"),
        ):
            assert get_backend(*operands) == expected
            assert get_array_module(*operands) is (np if expected == "numpy" else jnp)
        assert get_backend() == configured


@pytest.mark.requires_jax
def test_explicit_array_backend_overrides_operands_without_changing_global_policy():
    """Explicit materialization choices share the ordinary backend interface."""
    import jax.numpy as jnp

    for configured in ("numpy", "jax"):
        with backend_context(configured):
            assert get_array_module(jnp.ones(1), backend=" NUMPY ") is np
            assert get_array_module(np.ones(1), backend="jax") is jnp
            assert get_array_module(jnp.ones(1)) is jnp
            assert get_backend() == configured


@pytest.mark.parametrize(
    "backend, error", [(False, TypeError), (np, TypeError), ("auto", ValueError)]
)
def test_explicit_array_backend_rejects_unsupported_requests(backend, error):
    with pytest.raises(error, match="backend"):
        get_array_module(backend=backend)


def test_readonly_numpy_array_owns_contiguous_metadata():
    """Cached metadata cannot be changed through its original array."""
    source = np.arange(12).reshape(3, 4).T
    owned = readonly_numpy_array(source, dtype=float)
    np.testing.assert_array_equal(owned, source)
    source[0, 0] = -1
    assert owned[0, 0] == 0.0
    assert owned.dtype == np.dtype(float)
    assert owned.flags.c_contiguous
    assert not owned.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        owned[0, 0] = 1.0


@pytest.mark.requires_jax
def test_readonly_numpy_array_is_an_explicit_host_boundary():
    """Metadata stays on NumPy even when the numerical backend is JAX."""
    import jax.numpy as jnp

    from kompe.math import backend_context

    with backend_context("jax"):
        owned = readonly_numpy_array(jnp.arange(3.0))
    assert isinstance(owned, np.ndarray)
    assert not owned.flags.writeable
    np.testing.assert_array_equal(owned, np.arange(3.0))


def _run_python(source, *, env=None):
    environment = os.environ.copy()
    environment.pop("KOMPE_USE_JAX", None)
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_importing_kompe_does_not_import_jax():
    """NumPy-only callers do not pay JAX's import or initialization cost."""
    result = _run_python(
        "import sys, numpy as np, kompe; "
        "from kompe.math import get_backend; "
        "assert get_backend() == get_backend(np.ones(1)) == 'numpy'; "
        "print('jax' in sys.modules)"
    )
    assert result.stdout.strip() == "False"


@pytest.mark.requires_jax
def test_querying_configured_backend_does_not_import_jax():
    """Reading configuration does not initialize the selected backend."""
    result = _run_python(
        "import sys; from kompe.math import get_backend; "
        "print(get_backend(), 'jax' in sys.modules)",
        env={"KOMPE_USE_JAX": "1"},
    )
    assert result.stdout.strip() == "jax False"


@pytest.mark.requires_jax
def test_enabling_backend_preserves_x64_setting():
    """Backend selection leaves application-wide JAX precision unchanged."""
    source = (
        "import jax; before = bool(jax.config.jax_enable_x64); "
        "from kompe.math import set_backend; set_backend('jax'); "
        "print(before, bool(jax.config.jax_enable_x64))"
    )
    result = _run_python(source, env={"JAX_ENABLE_X64": "0"})
    assert result.stdout.strip() == "False False"
