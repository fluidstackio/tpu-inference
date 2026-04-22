# SPDX-License-Identifier: Apache-2.0
"""Prefix-chained SHA256 hasher for the KV offload PoC.

The hasher produces one fixed-width integer hash per *chunk* of tokens,
chained across a request so that two requests sharing a prefix produce
identical chunk hashes for the shared portion.

Design:
  chunk_i      = tokens[i*CHUNK : (i+1)*CHUNK]
  prev_hash_0  = 0
  prev_hash_i  = hash(prev_hash_{i-1}, chunk_i)

Only full chunks are emitted; a trailing partial chunk (fewer than
CHUNK tokens) is not hashed by this module. The caller decides what to
do with the partial suffix (typically: prefill it normally).

128-bit truncation: SHA256 produces 256-bit output; we take the first
128 bits as an int. Collision probability is negligible for any
realistic cache-size * request-count product.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Iterator, Sequence

# 256-bit truncation to 128 bits is sufficient. Keep the name parameter so
# callers can see the contract in module-level constants.
_HASH_BITS = 128
_HEX_CHARS = _HASH_BITS // 4  # 32 hex chars -> 128 bits


def _hash_chunk(prev: int, chunk: Sequence[int]) -> int:
    """One step of the prefix chain. Pure function; no randomness."""
    h = hashlib.sha256()
    # Feed the previous hash as fixed-width bytes so the first chunk
    # (prev == 0) is still distinct from any other hash value.
    h.update(prev.to_bytes(_HASH_BITS // 8, "big", signed=False))
    # Feed token ids as 8-byte little-endian (tokens fit in int32 today,
    # int64 tomorrow; 8 bytes is safe forever).
    for t in chunk:
        h.update(int(t).to_bytes(8, "little", signed=False))
    return int(h.hexdigest()[:_HEX_CHARS], 16)


def iter_chunk_hashes(
    token_ids: Sequence[int],
    chunk_size: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield (start_idx, end_idx, prefix_hash) for each full chunk.

    Partial trailing chunks are not emitted.

    Example with chunk_size=4 and tokens=[a, b, c, d, e, f, g]:
      yields (0, 4, h1) and (4, 8 ... actually 4, 8 not reached, so only (0, 4, h1))
      The trailing [e, f, g] is partial and not hashed.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")
    total = len(token_ids)
    num_full = total // chunk_size

    prev = 0
    for i in range(num_full):
        start = i * chunk_size
        end = start + chunk_size
        chunk = token_ids[start:end]
        prev = _hash_chunk(prev, chunk)
        yield (start, end, prev)


def hash_prefix(token_ids: Sequence[int], chunk_size: int) -> list[int]:
    """Return the ordered list of chunk hashes for a prefix.

    Two requests whose first ``k * chunk_size`` tokens are identical
    will have identical ``hash_prefix(...)[:k]``.
    """
    return [h for _, _, h in iter_chunk_hashes(token_ids, chunk_size)]


def match_common_prefix(
    left_hashes: Sequence[int],
    right_hashes: Iterable[int],
) -> int:
    """Count how many leading chunk hashes are common to both sides.

    Used by a lookup: compare the request's computed chunk hashes
    against what's in the CPU store and stop at the first mismatch or
    the first hash the store doesn't have.
    """
    count = 0
    for left, right in zip(left_hashes, right_hashes):
        if left != right:
            return count
        count += 1
    return count
