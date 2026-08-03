"""Shared test capability markers."""

from __future__ import annotations

import pytest

from kompe.math import JAX_AVAILABLE


def pytest_collection_modifyitems(items):
    """Skip optional-backend tests when JAX is unavailable."""
    if JAX_AVAILABLE:
        return
    marker = pytest.mark.skip(reason="JAX is not installed.")
    for item in items:
        if item.get_closest_marker("requires_jax"):
            item.add_marker(marker)
