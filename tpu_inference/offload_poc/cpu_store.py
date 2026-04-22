# SPDX-License-Identifier: Apache-2.0
"""In-memory DRAM store for offloaded KV cache chunks.

The store maps an opaque chunk key (a prefix hash) to opaque chunk
payload (host-resident KV data). It implements:

  * LRU eviction when capacity is reached.
  * Pin-on-probe reference counting: a chunk that has been reported
    as a cache hit to the vLLM scheduler cannot be evicted until the
    worker has loaded it back to HBM. This prevents a subtle
    correctness hazard where:

        scheduler reports "N tokens already computed"     <- probe
           ... next iteration ...
        a different request's put triggers LRU eviction
        of our chunk
           ... worker tries to load ...
        chunk is gone -> forward pass runs on empty KV

The store is agnostic to the shape or dtype of the chunk payload.
Callers (the worker) stash arbitrary Python objects (typically a
``jax.Array`` backed by host pinned memory) under a key.

Concurrency: the store is not thread-safe. The connector's scheduler
and worker run in the same process, and vLLM serializes their
interaction around the engine step, so no internal locking is needed
for the PoC. A production version would revisit this.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class PoCCPUStore:
    """LRU-evicting map with probe/release reference counting.

    API:
        probe(key)   -> increment refcount, return payload if present
        release(key) -> decrement refcount (must pair with a probe)
        put(key, v)  -> insert; may evict the least-recently-used unpinned entry
        get(key)     -> fetch without touching refcount (for debug/metrics)
        __len__      -> current occupied entries
    """

    def __init__(self, max_entries: int):
        if max_entries <= 0:
            raise ValueError(
                f"max_entries must be positive; got {max_entries}")
        self._max = max_entries
        self._data: OrderedDict[Any, Any] = OrderedDict()
        # Refcount per key. A key in _refs with value > 0 is pinned
        # and cannot be evicted.
        self._refs: dict[Any, int] = {}

    # ---- capacity / query --------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def get(self, key: Any) -> Any | None:
        """Fetch without affecting LRU order or refcount. Returns None on miss."""
        return self._data.get(key)

    # ---- probe / release lifecycle -----------------------------------------

    def probe(self, key: Any) -> Any | None:
        """Record a probe and return the payload if present.

        Increments the refcount. The caller must pair this with a
        ``release(key)`` after the load has completed. A miss does not
        touch the refcount.
        """
        payload = self._data.get(key)
        if payload is None:
            return None
        # Touch LRU order on hit so frequently-used chunks stay warm.
        self._data.move_to_end(key)
        self._refs[key] = self._refs.get(key, 0) + 1
        return payload

    def release(self, key: Any) -> None:
        """Decrement the refcount. No-op if the key is unknown.

        An unknown key is tolerated silently: if the store was
        corrupted (shouldn't happen if probe/release are paired), we
        prefer to drop the release rather than crash the serving loop.
        The pin-on-probe invariant guarantees the key is still present
        during a correctly-paired probe/release, so unknown-key here
        means somebody called release without a prior probe.
        """
        n = self._refs.get(key)
        if n is None:
            return
        if n <= 1:
            self._refs.pop(key, None)
        else:
            self._refs[key] = n - 1

    # ---- insertion with LRU eviction ---------------------------------------

    def put(self, key: Any, payload: Any) -> bool:
        """Insert ``key -> payload``. Returns True if a new slot was consumed.

        If ``key`` already exists, its payload is replaced and LRU
        order updated. A re-put does not consume a new slot.

        If the store is at capacity, the least-recently-used *unpinned*
        entry is evicted to make room. If no unpinned entry exists
        (every slot is pinned), the put raises RuntimeError to prevent
        the caller from silently losing data.
        """
        if key in self._data:
            self._data[key] = payload
            self._data.move_to_end(key)
            return False

        if len(self._data) >= self._max:
            self._evict_one_lru_unpinned()

        self._data[key] = payload
        return True

    def _evict_one_lru_unpinned(self) -> None:
        """Evict the oldest entry with refcount == 0.

        Walks the LRU order from oldest to newest; removes the first
        entry that is not pinned. Raises if every entry is pinned
        (degenerate case: caller must grow the store or reduce
        concurrency).
        """
        for key in self._data:
            if self._refs.get(key, 0) == 0:
                self._data.pop(key)
                return
        raise RuntimeError(
            "PoCCPUStore is full and every entry is pinned; cannot evict. "
            "This indicates either a probe/release mismatch or a capacity "
            "misconfiguration (max_entries too small for in-flight load fanout)."
        )

    # ---- inspection --------------------------------------------------------

    def refcount(self, key: Any) -> int:
        """Return current refcount for a key (0 if unpinned or unknown)."""
        return self._refs.get(key, 0)

    def keys(self):
        """Iterate keys in LRU order (oldest first)."""
        return self._data.keys()
