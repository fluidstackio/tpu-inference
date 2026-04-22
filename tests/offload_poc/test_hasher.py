# SPDX-License-Identifier: Apache-2.0
"""Unit tests for offload_poc.hasher."""

import pytest

from tpu_inference.offload_poc.hasher import (
    hash_prefix,
    iter_chunk_hashes,
    match_common_prefix,
)


def test_determinism():
    """Same input -> same hashes, every time."""
    tokens = list(range(1000))
    a = hash_prefix(tokens, chunk_size=256)
    b = hash_prefix(tokens, chunk_size=256)
    assert a == b
    assert len(a) == 1000 // 256  # 3 full chunks


def test_prefix_chain_property():
    """Identical prefixes produce identical leading hashes."""
    shared = list(range(600))
    req_a = shared + [10, 11, 12, 13, 14, 15]
    req_b = shared + [20, 21, 22, 23, 24, 25]

    ha = hash_prefix(req_a, chunk_size=256)
    hb = hash_prefix(req_b, chunk_size=256)

    # Both have 600 shared tokens = 2 full chunks of 256 (total 512).
    # After that, tokens at position 512..600 (88 tokens) are still shared,
    # but not enough for a 3rd full chunk. Each request then adds 6 more,
    # giving 94 trailing tokens -- still less than one more chunk.
    # Only full shared chunks should match.
    assert ha[0] == hb[0]
    assert ha[1] == hb[1]
    # No 3rd chunk was emitted (partial).
    assert len(ha) == 2
    assert len(hb) == 2


def test_partial_trailing_chunk_not_emitted():
    """A partial final chunk is silently dropped."""
    # 3 full chunks + 50 trailing tokens.
    tokens = list(range(256 * 3 + 50))
    hashes = hash_prefix(tokens, chunk_size=256)
    assert len(hashes) == 3


def test_chunk_alignment():
    """iter_chunk_hashes reports correct (start, end) for each chunk."""
    tokens = list(range(768))  # exactly 3 chunks of 256
    triples = list(iter_chunk_hashes(tokens, chunk_size=256))
    assert [(s, e) for s, e, _ in triples] == [(0, 256), (256, 512), (512, 768)]


def test_prev_hash_zero_is_consistent():
    """The initial prev-hash (0) must be deterministic.

    This guards against a bug where the first chunk's hash depends on
    uninitialised state.
    """
    tokens = list(range(256))
    h1 = hash_prefix(tokens, chunk_size=256)
    h2 = hash_prefix(tokens, chunk_size=256)
    assert h1 == h2
    # And the value should be nonzero (hash of something is not 0 by chance).
    assert h1[0] != 0


def test_different_tokens_different_hashes():
    """Single-token difference in first chunk changes every subsequent hash."""
    a = list(range(1000))
    b = list(range(1000))
    b[0] = 999_999  # perturb first token only
    ha = hash_prefix(a, chunk_size=256)
    hb = hash_prefix(b, chunk_size=256)
    assert ha[0] != hb[0]
    # Chain property: first diff propagates forward.
    for x, y in zip(ha, hb):
        assert x != y


def test_different_chunk_sizes_produce_different_hashes():
    """Same tokens under a different chunk_size are a different cache namespace."""
    tokens = list(range(1024))
    h256 = hash_prefix(tokens, chunk_size=256)
    h512 = hash_prefix(tokens, chunk_size=512)
    # No hash value from one should appear in the other (very high probability).
    assert not (set(h256) & set(h512))


def test_match_common_prefix_exact():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    assert match_common_prefix(a, b) == 5


def test_match_common_prefix_diverge():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 99, 4, 5]
    assert match_common_prefix(a, b) == 2


def test_match_common_prefix_empty():
    assert match_common_prefix([], []) == 0
    assert match_common_prefix([1, 2], []) == 0
    assert match_common_prefix([], [1, 2]) == 0


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        hash_prefix([1, 2, 3], chunk_size=0)
    with pytest.raises(ValueError):
        hash_prefix([1, 2, 3], chunk_size=-1)


def test_empty_tokens_yields_no_hashes():
    assert hash_prefix([], chunk_size=256) == []


def test_fewer_tokens_than_chunk_size():
    """If we don't even have one full chunk, emit nothing."""
    assert hash_prefix([1, 2, 3], chunk_size=256) == []
