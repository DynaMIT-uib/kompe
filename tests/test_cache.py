"""Tests for bounded reusable numerical work."""

import pytest

from kompe.cache import BoundedCache


def test_bounded_cache_builds_once_and_updates_recency():
    cache = BoundedCache(2)
    builds = []

    first = cache.get_or_create("first", lambda: builds.append("first") or object())
    cache.get_or_create("second", lambda: builds.append("second") or object())

    assert cache.get_or_create("first", lambda: pytest.fail("cache hit was rebuilt")) is first

    cache.get_or_create("third", lambda: builds.append("third") or object())

    assert builds == ["first", "second", "third"]
    assert cache.get("second") is None
    assert next(iter(cache.values())) is first


def test_bounded_cache_can_store_none_and_replace_values():
    cache = BoundedCache(1)
    cache.get_or_create("none", lambda: None)

    assert cache.get_or_create("none", lambda: pytest.fail("cached None was rebuilt")) is None

    replacement = object()
    cache.store("none", replacement)
    assert cache.get("none") is replacement
    cache.clear()
    assert len(cache) == 0


@pytest.mark.parametrize("max_size", [0, -1])
def test_bounded_cache_requires_positive_size(max_size):
    with pytest.raises(ValueError, match="positive"):
        BoundedCache(max_size)
