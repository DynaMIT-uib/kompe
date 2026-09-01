"""Tests for optional-backend import and configuration policy."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from kompe.math import readonly_numpy_array


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
    result = _run_python("import sys, kompe; print('jax' in sys.modules)")
    assert result.stdout.strip() == "False"


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
