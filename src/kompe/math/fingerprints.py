"""Deterministic content fingerprints for numerical cache identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

_DEFAULT_DIGEST_SIZE = 16


def array_fingerprint(values, *, dtype=None) -> tuple[tuple[int, ...], str, str] | None:
    """Return exact shape, dtype, and content identity.

    ``None`` is returned for values that cannot be represented as a
    non-object NumPy array. An explicit ``dtype`` may be used when the
    caller intentionally defines a quantized cache identity.
    """
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError):
        return None
    if array.dtype.hasobject:
        return None

    shape = tuple(int(size) for size in array.shape)
    canonical = np.ascontiguousarray(array)
    digest = hashlib.blake2b(
        canonical.view(np.uint8), digest_size=_DEFAULT_DIGEST_SIZE
    ).hexdigest()
    return shape, canonical.dtype.str, digest


def _canonical_content(value):
    """Return a JSON-compatible, type-preserving content description."""
    if isinstance(value, np.ndarray):
        fingerprint = array_fingerprint(value)
        if fingerprint is None:
            raise TypeError("Object arrays cannot be fingerprinted.")
        shape, dtype, digest = fingerprint
        return {"array": {"shape": list(shape), "dtype": dtype, "digest": digest}}
    if isinstance(value, np.generic):
        return _canonical_content(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Non-finite floats cannot be fingerprinted.")
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes": hashlib.blake2b(value, digest_size=_DEFAULT_DIGEST_SIZE).hexdigest()}
    if isinstance(value, tuple):
        return {"tuple": [_canonical_content(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_canonical_content(item) for item in value]}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Fingerprint mapping keys must be strings.")
        return {"mapping": {key: _canonical_content(item) for key, item in sorted(value.items())}}
    raise TypeError(f"Unsupported fingerprint value type: {type(value).__name__}.")


def content_fingerprint(value) -> str:
    """Return a deterministic digest for nested numerical metadata."""
    encoded = json.dumps(
        _canonical_content(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=_DEFAULT_DIGEST_SIZE).hexdigest()


__all__ = ["array_fingerprint", "content_fingerprint"]
