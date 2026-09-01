"""Small in-memory caches for reusable numerical work."""

from collections import OrderedDict
from collections.abc import Callable
from operator import index
from typing import Any


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


__all__ = ["BoundedCache"]
