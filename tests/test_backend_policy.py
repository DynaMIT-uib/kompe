"""Tests for optional-backend import and configuration policy."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from kompe.math import JAX_AVAILABLE


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


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_enabling_backend_loads_jax_without_enabling_x64():
    """Kompe leaves process-wide JAX precision policy to the application."""
    source = (
        "import jax; before = bool(jax.config.jax_enable_x64); "
        "from kompe.math import set_backend; set_backend('jax'); "
        "print(before, bool(jax.config.jax_enable_x64))"
    )
    result = _run_python(source, env={"JAX_ENABLE_X64": "0"})
    assert result.stdout.strip() == "False False"
