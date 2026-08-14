"""Array backend helpers for kompe math code.

This module centralizes the optional JAX acceleration policy used by
linear maps, tensor operations, least-squares solvers, and simulation
code. The active backend is controlled through ``KOMPE_USE_JAX`` and
can also be changed programmatically with ``set_backend`` or
``use_jax``.
"""

from __future__ import annotations

import os
import types
from contextlib import contextmanager
from functools import wraps
from importlib.util import find_spec
from threading import RLock
from typing import Any

import numpy as _np

JAX_AVAILABLE = find_spec("jax") is not None
_jax_namespace: types.ModuleType | None = None
_jax_array_type: tuple[type, ...] = ()
_jax_device_put = None
_jax_jit = None
_jax_vmap = None
_jax_import_lock = RLock()


def _load_jax() -> types.ModuleType:
    """Import JAX on first use without changing process-wide configuration."""
    global _jax_array_type, _jax_device_put, _jax_jit, _jax_namespace, _jax_vmap

    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed; cannot enable JAX backend.")
    if _jax_namespace is not None:
        return _jax_namespace

    with _jax_import_lock:
        if _jax_namespace is not None:
            return _jax_namespace
        try:
            import jax
            import jax.numpy as jnp
            from jax import Array as JaxArray
        except ImportError as exc:  # pragma: no cover - broken optional install
            raise RuntimeError("JAX is installed but could not be imported.") from exc

        _jax_namespace = jnp
        _jax_device_put = jax.device_put
        _jax_jit = jax.jit
        _jax_vmap = jax.vmap
        _jax_array_type = (JaxArray,)
        return _jax_namespace


def _is_jax_array(array: Any) -> bool:
    """Recognize JAX values without importing JAX for NumPy-only callers."""
    array_module = type(array).__module__.split(".", maxsplit=1)[0]
    if array_module not in {"jax", "jaxlib"}:
        return False
    _load_jax()
    return isinstance(array, _jax_array_type)


def _env_flag(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


_USE_JAX = JAX_AVAILABLE and _env_flag(os.environ.get("KOMPE_USE_JAX"))


def use_jax(flag: bool | None = None) -> bool:
    """Query or set whether JAX should be used."""
    global _USE_JAX
    if flag is not None:
        if flag and not JAX_AVAILABLE:
            raise RuntimeError("JAX is not installed; cannot enable JAX backend.")
        if flag:
            _load_jax()
        _USE_JAX = bool(flag)
    return bool(_USE_JAX and JAX_AVAILABLE)


def set_backend(backend: str | bool | None) -> str:
    """Set the active array backend."""
    if isinstance(backend, bool):
        target = backend
    elif backend is None:
        target = use_jax()
    elif isinstance(backend, str):
        normalized = backend.strip().lower()
        if normalized in {"jax"}:
            target = True
        elif normalized == "numpy":
            target = False
        elif normalized == "auto":
            target = use_jax()
        else:
            raise ValueError(f"Unknown backend '{backend}'. Expected 'numpy', 'jax', or 'auto'.")
    else:
        raise TypeError("backend must be a string, boolean, or None.")

    use_jax(target)
    os.environ["KOMPE_USE_JAX"] = "1" if target else "0"
    return "jax" if target else "numpy"


@contextmanager
def backend_context(backend: str | bool | None):
    """Temporarily set the active array backend inside a context."""
    previous_backend = use_jax()
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


def get_array_module(*arrays: Any) -> types.ModuleType:
    """Return the active array module.

    Explicit JAX inputs take precedence over the global backend setting.
    """
    if arrays:
        for array in arrays:
            if _is_jax_array(array):
                return _load_jax()
    return _load_jax() if use_jax() else _np


def to_jax(array: Any) -> Any:
    """Convert ``array`` to a JAX device array when JAX is enabled."""
    if not use_jax():
        return array
    _load_jax()
    return _jax_device_put(array)


def block_until_ready(array: Any) -> Any:
    """Synchronize a JAX array before handing work to NumPy/SciPy."""
    if isinstance(array, tuple):
        for item in array:
            block_until_ready(item)
        return array
    if isinstance(array, list):
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


def block_after_jax_linalg(array: Any) -> Any:
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
    return _np.asarray(array)


def asarray(array: Any, dtype: Any = None) -> Any:
    """Backend-aware ``asarray`` helper."""
    module = get_array_module(array)
    return module.asarray(array, dtype=dtype) if dtype is not None else module.asarray(array)


def _identity_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def jit(func=None, *jit_args, **jit_kwargs):
    """Wrap ``jax.jit`` and no-op when JAX is disabled."""
    if not JAX_AVAILABLE:
        if func is None:
            return _identity_decorator
        return func

    def decorator(fn):
        if use_jax():
            _load_jax()
            return _jax_jit(fn, *jit_args, **jit_kwargs)
        return fn

    if func is None:
        return decorator
    return decorator(func)


def vmap(func=None, *vmap_args, **vmap_kwargs):
    """Wrap ``jax.vmap`` and require the JAX backend."""
    if not (JAX_AVAILABLE and use_jax()):
        raise RuntimeError("JAX vmap requested but the JAX backend is not enabled.")
    _load_jax()

    if func is None:
        return lambda fn: _jax_vmap(fn, *vmap_args, **vmap_kwargs)

    return _jax_vmap(func, *vmap_args, **vmap_kwargs)


__all__ = [
    "JAX_AVAILABLE",
    "asarray",
    "backend_context",
    "block_after_jax_linalg",
    "block_until_ready",
    "get_array_module",
    "jit",
    "set_backend",
    "to_jax",
    "to_numpy",
    "use_jax",
    "vmap",
]
