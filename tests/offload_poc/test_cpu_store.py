# SPDX-License-Identifier: Apache-2.0
"""Unit tests for offload_poc.cpu_store.

Focus: the pin-on-probe invariant. Without this, evictions between
scheduler probe and worker load would silently corrupt KV state.
"""

import pytest

from tpu_inference.offload_poc.cpu_store import PoCCPUStore


def test_basic_put_get():
    s = PoCCPUStore(max_entries=4)
    assert s.put("a", 1) is True
    assert s.get("a") == 1
    assert len(s) == 1


def test_put_overwrite_does_not_consume_slot():
    s = PoCCPUStore(max_entries=2)
    s.put("a", 1)
    s.put("b", 2)
    # Overwrite; should not count as a new slot.
    assert s.put("a", 99) is False
    assert len(s) == 2
    assert s.get("a") == 99


def test_lru_eviction():
    s = PoCCPUStore(max_entries=3)
    s.put("a", 1)
    s.put("b", 2)
    s.put("c", 3)
    # Next put should evict "a" (least recently used).
    s.put("d", 4)
    assert "a" not in s
    assert set(s.keys()) == {"b", "c", "d"}


def test_lru_touch_on_probe():
    """A probe must count as a recent access, protecting against eviction."""
    s = PoCCPUStore(max_entries=3)
    s.put("a", 1)
    s.put("b", 2)
    s.put("c", 3)
    # Probe and release "a" so it's fresh AND unpinned.
    assert s.probe("a") == 1
    s.release("a")
    # Now insert "d"; the oldest unpinned entry should be "b", not "a".
    s.put("d", 4)
    assert "a" in s
    assert "b" not in s


def test_pin_prevents_eviction():
    """Pinned entries must not be evicted even if they are LRU."""
    s = PoCCPUStore(max_entries=3)
    s.put("a", 1)
    s.put("b", 2)
    s.put("c", 3)
    # Pin "a" (the LRU slot) via probe without release.
    assert s.probe("a") == 1
    # Insert "d": should evict "b" (next LRU unpinned), NOT "a".
    s.put("d", 4)
    assert "a" in s
    assert "b" not in s
    assert s.refcount("a") == 1


def test_all_pinned_raises_on_put():
    """If every slot is pinned, put must refuse rather than silently lose data."""
    s = PoCCPUStore(max_entries=2)
    s.put("a", 1)
    s.put("b", 2)
    s.probe("a")
    s.probe("b")
    with pytest.raises(RuntimeError):
        s.put("c", 3)


def test_probe_miss_does_not_pin():
    """A probe on an absent key must not leave a phantom refcount."""
    s = PoCCPUStore(max_entries=2)
    assert s.probe("nope") is None
    assert s.refcount("nope") == 0


def test_probe_release_pair():
    s = PoCCPUStore(max_entries=2)
    s.put("a", 1)
    s.probe("a")
    assert s.refcount("a") == 1
    s.release("a")
    assert s.refcount("a") == 0


def test_multiple_probes_stack():
    """Two concurrent probes must both pin until both release."""
    s = PoCCPUStore(max_entries=2)
    s.put("a", 1)
    s.probe("a")
    s.probe("a")
    assert s.refcount("a") == 2
    s.release("a")
    assert s.refcount("a") == 1  # still pinned
    s.release("a")
    assert s.refcount("a") == 0


def test_release_unknown_key_is_noop():
    """Release of an unknown key must not raise (tolerates mismatched calls)."""
    s = PoCCPUStore(max_entries=2)
    s.release("never-probed")  # no-op, no exception


def test_get_does_not_touch_lru_or_refs():
    """get() is for debug/metrics; must not affect invariants."""
    s = PoCCPUStore(max_entries=3)
    s.put("a", 1)
    s.put("b", 2)
    s.put("c", 3)
    # Plain get on "a"; should NOT protect it from eviction.
    assert s.get("a") == 1
    assert s.refcount("a") == 0
    s.put("d", 4)
    assert "a" not in s  # still evicted


def test_zero_capacity_rejected():
    with pytest.raises(ValueError):
        PoCCPUStore(max_entries=0)
    with pytest.raises(ValueError):
        PoCCPUStore(max_entries=-1)


def test_contains_and_len():
    s = PoCCPUStore(max_entries=4)
    assert "a" not in s
    assert len(s) == 0
    s.put("a", 1)
    assert "a" in s
    assert len(s) == 1
