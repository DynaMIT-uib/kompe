"""In-memory and persistent caches for reusable numerical work."""

from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from operator import index
from pathlib import Path
from typing import Any

import numpy as np

from kompe.math.fingerprints import content_fingerprint


class BoundedCache:
    """Retain a fixed number of recently used values in memory."""

    def __init__(self, max_size):
        try:
            max_size = index(max_size)
        except TypeError as exc:
            raise TypeError("max_size must be an integer.") from exc
        if max_size < 1:
            raise ValueError("max_size must be positive.")
        self.max_size = max_size
        self._values = OrderedDict()

    def get(self, key, default=None):
        """Return ``key`` and mark it as recently used."""
        if key not in self._values:
            return default
        value = self._values.pop(key)
        self._values[key] = value
        return value

    def store(self, key, value):
        """Store ``value`` and discard the least recently used excess entry."""
        self._values.pop(key, None)
        self._values[key] = value
        if len(self._values) > self.max_size:
            self._values.popitem(last=False)

    def get_or_create(self, key, build: Callable[[], Any]):
        """Return the cached value for ``key``, building it when absent."""
        if key in self._values:
            return self.get(key)
        value = build()
        self.store(key, value)
        return value

    def clear(self):
        """Discard all retained values."""
        self._values.clear()

    def values(self):
        """Return retained values from least to most recently used."""
        return self._values.values()

    def __len__(self):
        """Return the number of retained values."""
        return len(self._values)


_CACHE_FORMAT_VERSION = 1


def _safe_category(category: str) -> str:
    """Return a path-safe cache category."""
    category = str(category)
    if (
        not category
        or category in {".", ".."}
        or "/" in category
        or "\\" in category
        or Path(category).name != category
    ):
        raise ValueError(f"Cache category must be one path-safe name, got {category!r}.")
    return category


class PersistentArrayCache:
    """Store immutable NumPy arrays under exact content-derived keys.

    Cache entries are optional performance artifacts. Callers supply
    deterministic builders, so missing entries can be reconstructed.
    Arrays load read-only through NumPy memory mapping.
    """

    def __init__(self, directory: str | os.PathLike[str]):
        """Bind the cache to ``directory``."""
        self.directory = Path(directory).expanduser().resolve()

    def _paths(self, category: str, identity) -> tuple[Path, Path, str]:
        """Return array, manifest, and digest for one cache entry."""
        category = _safe_category(category)
        digest = content_fingerprint(
            {
                "cache_format_version": _CACHE_FORMAT_VERSION,
                "category": category,
                "identity": identity,
            }
        )
        entry_directory = self.directory / category
        return (entry_directory / f"{digest}.npy", entry_directory / f"{digest}.json", digest)

    @staticmethod
    def _load(array_path: Path, manifest_path: Path, digest: str) -> np.ndarray | None:
        """Load and validate one complete entry, or return ``None``."""
        if not array_path.is_file() or not manifest_path.is_file():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid array-cache entry {array_path}. Remove the entry and retry."
            ) from exc

        expected = {
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "key": digest,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "nbytes": int(array.nbytes),
        }
        if manifest != expected:
            raise RuntimeError(
                f"Array-cache manifest does not match {array_path}. Remove the entry and retry."
            )
        return array

    @staticmethod
    def _write_json_atomically(path: Path, value: dict) -> None:
        """Write one JSON file atomically."""
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", suffix=".json", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temp_name, path)
        except Exception:
            Path(temp_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_array_atomically(path: Path, array: np.ndarray) -> None:
        """Write one NPY array atomically."""
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", suffix=".npy", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.save(stream, array, allow_pickle=False)
            os.replace(temp_name, path)
        except Exception:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def get_or_create(
        self, category: str, identity, builder: Callable[[], np.ndarray]
    ) -> np.ndarray:
        """Return a validated cached array, building it when absent."""
        array_path, manifest_path, digest = self._paths(category, identity)
        cached = self._load(array_path, manifest_path, digest)
        if cached is not None:
            return cached

        # Preserve an already contiguous C or Fortran layout. In
        # particular, copying a large LAPACK factor merely to change
        # its memory order can double peak construction memory.
        built = np.asarray(builder())
        if built.dtype.hasobject:
            raise TypeError("PersistentArrayCache cannot persist object arrays.")

        array_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_array_atomically(array_path, built)
        self._write_json_atomically(
            manifest_path,
            {
                "cache_format_version": _CACHE_FORMAT_VERSION,
                "key": digest,
                "shape": list(built.shape),
                "dtype": built.dtype.str,
                "nbytes": int(built.nbytes),
            },
        )
        cached = self._load(array_path, manifest_path, digest)
        if cached is None:  # pragma: no cover - guarded by the writes above
            raise RuntimeError(f"Failed to publish array-cache entry {array_path}.")
        return cached


__all__ = ["BoundedCache", "PersistentArrayCache"]
