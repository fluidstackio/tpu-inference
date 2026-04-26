# SPDX-License-Identifier: Apache-2.0
"""JIT gather / scatter ops for KV cache offload.

These functions are cherry-picked verbatim from the cpu-offloading
feature branch at `origin/cpu-offloading/dev-merge-0401` of
tpu-inference, file `tpu_inference/offload/utils.py` (PR #1163),
authored by the Google TPU Offload team. Only the four functions
below are taken; bucketing, hashing, and integration utilities are
intentionally left behind.

Source provenance:
    origin/cpu-offloading/dev-merge-0401:tpu_inference/offload/utils.py

Functions:
    stack_kv_cache_cross_layers(kv_caches, block_ids, num_blocks)
        Gather: collect non-contiguous HBM blocks across all layers
        into a stacked contiguous buffer. Returns the (re-donated)
        kv_caches and a list of per-block slices.

    pre_update_kv_caches(block_indices, mesh, replicated_sharding)
        Prepare the index arrays consumed by update_kv_caches.

    update_kv_caches(...)
        Scatter: write gathered blocks back into the KV caches at
        arbitrary non-contiguous positions, using the in-tree
        ``multi_layer_copy`` Pallas DMA kernel.

    update_kv_caches_one(kv_caches, stacked_blocks, block_indices,
                        mesh, replicated_sharding)
        Thin wrapper that calls ``pre_update_kv_caches`` followed by
        ``update_kv_caches``.

Notes on buffer donation:
    Both JIT-compiled functions declare ``donate_argnames=('kv_caches',)``.
    The caller MUST reassign the returned kv_caches handle to avoid
    using an invalidated donation. The connector threads this through
    ``runner.kv_caches`` after every save/load.
"""

from __future__ import annotations

import functools
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

from tpu_inference.distributed import kv_transfer


@functools.partial(
    jax.jit,
    static_argnames=['num_blocks'],
    donate_argnames=('kv_caches', ),
)
def stack_kv_cache_cross_layers(
    kv_caches: List[jax.Array],
    block_ids: jax.Array,
    num_blocks: int,
) -> Tuple[List[jax.Array], List[jax.Array]]:
    """Gather KV cache blocks identified by ``block_ids`` across all layers.

    Uses jax.tree.map to apply the gather across the per-layer list of
    KV cache arrays.
    """

    def _gather_blocks(layer_kv_cache):
        return layer_kv_cache.at[block_ids].get()

    gathered_kv_layers = jax.tree.map(_gather_blocks, kv_caches)
    stacked_blocks = jnp.stack(gathered_kv_layers, axis=1)

    # Split along axis=0 into individual blocks; num_blocks == len(block_ids).
    split_blocks = jnp.split(stacked_blocks,
                             indices_or_sections=num_blocks,
                             axis=0)

    kv_caches = jax.lax.optimization_barrier(kv_caches)
    return kv_caches, split_blocks


def pre_update_kv_caches(
    block_indices: List[int],
    mesh: Mesh,
    replicated_sharding: PartitionSpec | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the offset / chunk-size / num-chunks arrays for a scatter.

    Returned arrays are device-resident on the replicated sharding.
    """
    num_blocks = len(block_indices)
    src_offsets = jnp.arange(num_blocks, dtype=jnp.int32)
    dest_offsets = jnp.array(block_indices, dtype=jnp.int32)
    chunk_sizes = jnp.ones(num_blocks, dtype=jnp.int32)
    num_chunks = jnp.array([num_blocks], dtype=jnp.int32)

    if replicated_sharding is None:
        replicated_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(), memory_kind='device')
    src_offsets = jax.device_put(src_offsets, replicated_sharding)
    dest_offsets = jax.device_put(dest_offsets, replicated_sharding)
    chunk_sizes = jax.device_put(chunk_sizes, replicated_sharding)
    num_chunks = jax.device_put(num_chunks, replicated_sharding)

    return src_offsets, dest_offsets, chunk_sizes, num_chunks


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "src_sharding_spec",
        "dest_sharding_spec",
        "replicated_sharding_spec",
    ),
    donate_argnames=("kv_caches", ),
)
def update_kv_caches(
    kv_caches: List[jax.Array],
    stacked_blocks: List[jax.Array],
    src_offsets: jax.Array,
    dest_offsets: jax.Array,
    chunk_sizes: jax.Array,
    num_chunks: jax.Array,
    mesh,
    src_sharding_spec,
    dest_sharding_spec,
    replicated_sharding_spec,
) -> List[jax.Array]:
    """Scatter gathered blocks into KV caches at arbitrary positions.

    Args:
      kv_caches: List of original KV caches for each layer.
      stacked_blocks: List of gathered blocks, each with shape
        ``(1, num_layers, ...)``.
      src_offsets, dest_offsets, chunk_sizes, num_chunks: Index
        metadata from ``pre_update_kv_caches``.

    Returns:
      List of updated KV caches for each layer.
    """
    concatenated_blocks = jnp.concatenate(stacked_blocks, axis=0)
    layer_slices_tuple = jnp.unstack(concatenated_blocks, axis=1)
    layer_slices_list = list(layer_slices_tuple)

    # multi_layer_copy cherry-picked from PR #2026 does not take mesh /
    # sharding spec kwargs; those were added in a later revision. The
    # function derives sharding from the input arrays themselves.
    output = kv_transfer.multi_layer_copy(
        src_array=layer_slices_list,
        dest_array=kv_caches,
        src_offsets=src_offsets,
        dest_offsets=dest_offsets,
        chunk_sizes=chunk_sizes,
        num_chunks=num_chunks,
    )
    return output


def update_kv_caches_one(
    kv_caches: List[jax.Array],
    stacked_blocks: List[jax.Array],
    block_indices: List[int],
    mesh: Mesh,
    replicated_sharding: jax.sharding.NamedSharding | None = None,
) -> List[jax.Array]:
    """Convenience wrapper: prepare indices, then scatter.

    ``replicated_sharding`` is a NamedSharding (not a PartitionSpec).
    Its .spec is passed to ``update_kv_caches`` as the replicated
    sharding spec.
    """
    src_offsets, dest_offsets, chunk_sizes, num_chunks = pre_update_kv_caches(
        block_indices, mesh, replicated_sharding)
    return update_kv_caches(
        kv_caches,
        stacked_blocks,
        src_offsets,
        dest_offsets,
        chunk_sizes,
        num_chunks,
        mesh,
        kv_caches[0].sharding.spec,
        kv_caches[0].sharding.spec,
        replicated_sharding.spec,
    )
