"""Shared test capability markers."""

from __future__ import annotations

import os

import pytest

# Numerical regression tests request double precision explicitly. Kompe itself
# leaves this process-wide JAX policy to its caller.
os.environ.setdefault("JAX_ENABLE_X64", "1")

from kompe.math import JAX_AVAILABLE


def pytest_collection_modifyitems(items):
    """Skip optional-backend tests when JAX is unavailable."""
    if JAX_AVAILABLE:
        return
    marker = pytest.mark.skip(reason="JAX is not installed.")
    for item in items:
        if item.get_closest_marker("requires_jax"):
            item.add_marker(marker)
