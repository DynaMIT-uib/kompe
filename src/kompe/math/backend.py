"""Array backend helpers for kompe math code.

This module centralizes the optional JAX acceleration policy used by
linear maps, tensor operations, least-squares solvers, and simulation
code. The active backend is controlled through ``KOMPE_USE_JAX`` and
can also be changed programmatically with ``set_backend``.
"""

from __future__ import annotations

import os
import types
from contextlib import contextmanager
from importlib.util import find_spec
from threading import RLock
from typing import Any

import numpy as _np

JAX_AVAILABLE = find_spec("jax") is not None
_jax_namespace: types.ModuleType | None = None
_jax_array_type: tuple[type, ...] = ()
_jax_import_lock = RLock()


def _load_jax() -> types.ModuleType:
    """Import JAX on first use without changing process-wide configuration."""
    global _jax_array_type, _jax_namespace

    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed; cannot enable JAX backend.")
    if _jax_namespace is not None:
        return _jax_namespace

    with _jax_import_lock:
        if _jax_namespace is not None:
            return _jax_namespace
        try:
            import jax.numpy as jnp
            from jax import Array as JaxArray
        except ImportError as exc:  # pragma: no cover - broken optional install
            raise RuntimeError("JAX is installed but could not be imported.") from exc

        _jax_namespace = jnp
        _jax_array_type = (JaxArray,)
        return _jax_namespace


def _is_jax_array(array: Any) -> bool:
    """Recognize JAX values without importing JAX for NumPy-only callers."""
    array_module = type(array).__module__.split(".", maxsplit=1)[0]
    if array_module not in {"jax", "jaxlib"}:
        return False
    _load_jax()
    return isinstance(array, _jax_array_type)


_USE_JAX = JAX_AVAILABLE and os.environ.get("KOMPE_USE_JAX", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def jax_enabled() -> bool:
    """Return whether JAX is the active array backend."""
    return bool(_USE_JAX and JAX_AVAILABLE)


def get_backend(*arrays: Any) -> str:
    """Return the operand-implied or configured backend name.

    With no operands, report the configured backend without importing
    JAX. Otherwise, follow ``get_array_module(*arrays)``.
    """
    if arrays:
        return "numpy" if get_array_module(*arrays) is _np else "jax"
    return "jax" if jax_enabled() else "numpy"


def set_backend(backend: str | bool | None) -> str:
    """Set the active array backend."""
    global _USE_JAX

    if isinstance(backend, bool):
        target = backend
    elif backend is None:
        target = jax_enabled()
    elif isinstance(backend, str):
        normalized = backend.strip().lower()
        if normalized in {"jax"}:
            target = True
        elif normalized == "numpy":
            target = False
        elif normalized == "auto":
            target = jax_enabled()
        else:
            raise ValueError(f"Unknown backend '{backend}'. Expected 'numpy', 'jax', or 'auto'.")
    else:
        raise TypeError("backend must be a string, boolean, or None.")

    if target and not JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed; cannot enable JAX backend.")
    if target:
        _load_jax()
    _USE_JAX = bool(target)
    os.environ["KOMPE_USE_JAX"] = "1" if target else "0"
    return "jax" if target else "numpy"


@contextmanager
def backend_context(backend: str | bool | None):
    """Temporarily set the active array backend inside a context."""
    previous_backend = get_backend()
    previous_env = os.environ.get("KOMPE_USE_JAX")
    active_backend = set_backend(backend)
    try:
        yield active_backend
    finally:
        set_backend(previous_backend)
        if previous_env is None:
            os.environ.pop("KOMPE_USE_JAX", None)
        else:
            os.environ["KOMPE_USE_JAX"] = previous_env


def get_array_module(*arrays: Any, backend: str | None = None) -> types.ModuleType:
    """Return the requested, operand-implied, or configured array module.

    An explicit ``backend`` takes precedence over operands. Otherwise,
    JAX inputs take precedence over the global backend setting.
    """
    if backend is not None:
        if not isinstance(backend, str):
            raise TypeError("backend must be None, 'numpy', or 'jax'.")
        normalized = backend.strip().lower()
        if normalized == "numpy":
            return _np
        if normalized == "jax":
            return _load_jax()
        raise ValueError(f"Unknown array backend {backend!r}. Use None, 'numpy', or 'jax'.")
    if arrays:
        for array in arrays:
            if _is_jax_array(array):
                return _load_jax()
    return _load_jax() if jax_enabled() else _np


def block_until_ready(array: Any) -> Any:
    """Synchronize a JAX array before handing work to NumPy/SciPy."""
    if isinstance(array, (tuple, list)):
        for item in array:
            block_until_ready(item)
        return array
    if isinstance(array, dict):
        for item in array.values():
            block_until_ready(item)
        return array
    block = getattr(array, "block_until_ready", None)
    if block is not None:
        block()
    return array


def synchronize_linalg_result(array: Any) -> Any:
    """Synchronize JAX CPU linear algebra before NumPy/OpenBLAS work.

    JAX CPU ``pinv``/``svd``/``solve`` may lower to asynchronous LAPACK
    FFI calls. On the conda-forge OpenBLAS/OpenMP stack, leaving such a
    call in flight has corrupted later threaded NumPy GEMM/tensordot
    results. These barriers keep the JAX acceleration, but make LAPACK
    calls explicit handoff points before NumPy can run.
    """
    return block_until_ready(array)


def to_numpy(array: Any) -> Any:
    """Convert ``array`` to a NumPy ``ndarray``."""
    if _is_jax_array(array):
        block_until_ready(array)
    return _np.asarray(array)


def readonly_numpy_array(values: Any, *, dtype=None) -> _np.ndarray:
    """Own a read-only CPU copy of metadata or cached analysis weights.

    This is an explicit host boundary, not an array-backend choice.
    """
    array = _np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


__all__ = [
    "JAX_AVAILABLE",
    "backend_context",
    "block_until_ready",
    "get_array_module",
    "get_backend",
    "jax_enabled",
    "readonly_numpy_array",
    "set_backend",
    "synchronize_linalg_result",
    "to_numpy",
]
