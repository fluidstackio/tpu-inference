# SPDX-License-Identifier: Apache-2.0
"""TPUOffloadConnector — MVP PoC for KV cache offload to host DRAM.

Implements vLLM's ``KVConnectorBase_V1`` for TPU, storing evicted
prefix KV in host pinned DRAM and reloading on a prefix-hash hit.

Design is per the RFC ``rfc-ext-kv-cache-mvp-poc.md``:

  * DRAM only (no NVMe, no remote, no RDMA).
  * Synchronous save and load; no async pipeline.
  * Single startup warmup compile (no bucketing).
  * Pin-on-probe in the CPU store for correctness under eviction.

The class splits into three concerns:

  TPUOffloadConnector          - thin facade that routes to Scheduler
                                 or Worker based on role.
  _SchedulerImpl                - prefix hashing, cache lookup,
                                 save/load spec construction.
  _WorkerImpl                   - JIT gather/scatter, DMA to/from host,
                                 CPU store read/write.

Shared between the two: ``_PoCConnectorMetadata``, the per-step
message passed from scheduler to worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import functools

import jax
import jax.numpy as jnp

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)


@functools.partial(jax.jit, donate_argnums=(0,))
def _scatter_one_layer(layer_kv: jax.Array, dst_ids: jax.Array,
                       src_blocks: jax.Array) -> jax.Array:
    """In-place per-layer KV scatter. ``layer_kv`` is donated so the
    write does not double-allocate a fresh copy of the full per-layer
    buffer (which can be multiple GB on 8B+ models). Kept for
    reference / debugging; the fast path uses _scatter_all_layers."""
    return layer_kv.at[dst_ids].set(src_blocks)


@functools.partial(jax.jit, donate_argnums=(0,))
def _scatter_all_layers(
    kv_caches: list,
    dst_ids: jax.Array,
    srcs: list,
) -> list:
    """Single-jit scatter for all layers at once. kv_caches is a pytree
    (list) of per-layer arrays, donated; srcs is a same-length list of
    per-layer (num_blocks, block_size, ...) source arrays. Returns a
    new list of updated layers. Fusing 32 scatters into one jit graph
    cuts per-layer dispatch/barrier overhead which otherwise dominates
    the LOAD path cost on an 8B model.

    Deprecated by `_load_all`: this variant requires the caller to do
    the per-layer `jnp.stack` outside the jit, which paid 32× dispatch
    overhead. Kept for reference."""
    return [kv.at[dst_ids].set(src) for kv, src in zip(kv_caches, srcs)]


@functools.partial(jax.jit, donate_argnums=(0,))
def _load_all(
    kv_caches: list,
    dst_ids: jax.Array,
    flat_blocks: list,
) -> list:
    """Fused LOAD (Option α): concatenate saved per-block chunks into a
    single (num_blocks, num_layers, ...) tensor, then scatter into
    ``kv_caches`` per layer — all inside one jit graph. This removes
    the 32 per-layer ``jnp.stack`` dispatches that dominated the
    previous ``_scatter_all_layers`` path (see
    poc-perf-bottleneck-and-fix-plan.md).

    Shapes:
      flat_blocks: list of ``(1, num_layers, block_size, H, 2, D)``
                   device arrays, one per block being loaded.
      kv_caches:   list[num_layers] of ``(num_total_blocks, block_size,
                   H, 2, D)`` per-layer arrays (donated).
      dst_ids:     int32 vector of destination block indices, length
                   equals len(flat_blocks).
    """
    # (num_blocks, num_layers, block_size, H, 2, D)
    stacked = jnp.concatenate(flat_blocks, axis=0)
    return [
        kv.at[dst_ids].set(stacked[:, layer_idx])
        for layer_idx, kv in enumerate(kv_caches)
    ]

from tpu_inference import envs
from tpu_inference.distributed import kv_transfer
from tpu_inference.logger import init_logger
from tpu_inference.offload_poc.cpu_store import PoCCPUStore
from tpu_inference.offload_poc.hasher import hash_prefix
from tpu_inference.offload_poc.ops import (
    pre_update_kv_caches,
    stack_kv_cache_cross_layers,
    update_kv_caches,
)

if TYPE_CHECKING:
    import torch
    from vllm.attention import AttentionMetadata
    from vllm.config import KVCacheConfig, VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


# -----------------------------------------------------------------------------
# Metadata passed from scheduler to worker on each engine step.
# -----------------------------------------------------------------------------


@dataclass
class _SaveSpec:
    """Instructions to save a contiguous run of newly-filled KV blocks."""

    req_id: str
    chunk_hashes: list[int]  # one hash per chunk to be saved
    src_block_ids: list[int]  # HBM block ids holding the chunks
    # src_block_ids length is chunk_hashes length * (chunk_size / block_size)


@dataclass
class _LoadSpec:
    """Instructions to load matched chunks back into freshly allocated blocks."""

    req_id: str
    chunk_hashes: list[int]  # hashes that hit the CPU store
    dst_block_ids: list[int]  # HBM blocks to populate


@dataclass
class _PoCConnectorMetadata(KVConnectorMetadata):
    saves: list[_SaveSpec] = field(default_factory=list)
    loads: list[_LoadSpec] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Shared state (singleton).
#
# KVConnectorFactory constructs two TPUOffloadConnector instances per engine:
# one with role=SCHEDULER, one with role=WORKER (see vllm/v1/core/sched/
# scheduler.py and vllm/distributed/kv_transfer/kv_transfer_state.py). They
# are distinct Python objects in the same process, so any DRAM state stored
# on ``self`` is split across the two. We put the DRAM store and the per-
# request pending dicts on a module-level singleton so both roles share it.
# -----------------------------------------------------------------------------


class _SharedState:
    def __init__(self, max_entries: int):
        self.cpu_store = PoCCPUStore(max_entries=max_entries)
        self.pending_saves: dict[str, _SaveSpec] = {}
        self.pending_loads: dict[str, _LoadSpec] = {}
        # Request ids whose save has completed; drained by get_finished.
        self.done_sending: set[str] = set()


_SHARED_STATE: _SharedState | None = None


def _get_shared_state() -> _SharedState:
    global _SHARED_STATE
    if _SHARED_STATE is None:
        _SHARED_STATE = _SharedState(
            max_entries=envs.TPU_OFFLOAD_NUM_CPU_CHUNKS)
        logger.info(
            "[offload_poc] created shared state max_entries=%d",
            envs.TPU_OFFLOAD_NUM_CPU_CHUNKS)
    return _SHARED_STATE


# -----------------------------------------------------------------------------
# Scheduler-side implementation.
# -----------------------------------------------------------------------------


class _SchedulerImpl:
    """Scheduler-side connector logic.

    Owns the CPU-side view of what is cached: a map from chunk hash to
    a cheap token (currently the hash itself) that the worker can use
    to read back from the CPU store. In the PoC the scheduler and
    worker run in the same process, so the worker's PoCCPUStore serves
    as the source of truth; the scheduler queries it via ``probe``.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self._state = _get_shared_state()
        self._cpu_store = self._state.cpu_store
        self._chunk_size = envs.TPU_OFFLOAD_CHUNK_SIZE
        # pending_saves / pending_loads live on the shared state so that
        # the worker-side can drain and update them from a different
        # connector instance.
        self._pending_saves = self._state.pending_saves
        self._pending_loads = self._state.pending_loads

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Only consider tokens beyond what vLLM already has in HBM APC.
        tokens = request.prompt_token_ids
        logger.info(
            "[offload_poc] get_num_new_matched_tokens req=%s num_prompt=%d "
            "num_computed=%d store_len=%d",
            request.request_id, len(tokens), num_computed_tokens,
            len(self._cpu_store))
        if num_computed_tokens >= len(tokens):
            return 0, False

        # Hash the full prompt in chunks, then count the leading prefix
        # whose chunks are already in the CPU store. Probe each (pin-
        # on-probe) so the chunks cannot be evicted before load.
        all_hashes = hash_prefix(tokens, self._chunk_size)
        if not all_hashes:
            return 0, False

        matched_hashes: list[int] = []
        for h in all_hashes:
            if self._cpu_store.probe(h) is None:
                break
            matched_hashes.append(h)

        # Discard tokens that vLLM has already accounted for (APC hit).
        already_in_chunks = num_computed_tokens // self._chunk_size
        if already_in_chunks >= len(matched_hashes):
            # vLLM already covers everything we could have offered; release pins.
            for h in matched_hashes:
                self._cpu_store.release(h)
            return 0, False

        new_matched_hashes = matched_hashes[already_in_chunks:]
        # Release pins we won't actually use.
        for h in matched_hashes[:already_in_chunks]:
            self._cpu_store.release(h)

        # Remember what we pinned so we can build the load spec later.
        self._pending_loads[request.request_id] = _LoadSpec(
            req_id=request.request_id,
            chunk_hashes=new_matched_hashes,
            dst_block_ids=[],  # filled in update_state_after_alloc
        )

        num_matched_tokens = len(new_matched_hashes) * self._chunk_size
        # Synchronous load (second element False) — we load inside
        # start_load_kv, which runs before the forward pass.
        return num_matched_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        logger.info(
            "[offload_poc] update_state_after_alloc req=%s ext_tokens=%d",
            request.request_id, num_external_tokens)
        if num_external_tokens <= 0:
            return
        spec = self._pending_loads.get(request.request_id)
        if spec is None:
            return
        # Pull the freshly allocated block ids for the matched region.
        # The PoC assumes chunk_size is a multiple of the KV block size
        # so each chunk maps to an integer number of blocks.
        # We take only as many block ids as we need for the matched chunks.
        needed_blocks = self._num_blocks_for_chunks(len(spec.chunk_hashes))
        allocated_block_ids = _flatten_block_ids(blocks)
        spec.dst_block_ids = allocated_block_ids[:needed_blocks]

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = _PoCConnectorMetadata()

        # Emit loads queued by update_state_after_alloc for new requests.
        # Pending_loads entries survive until the worker confirms via
        # release-on-load-complete (see _WorkerImpl.start_load_kv).
        for spec in list(self._pending_loads.values()):
            if spec.dst_block_ids:  # only ready ones
                meta.loads.append(spec)

        # Saves are queued by request_finished. We keep them in
        # pending_saves until the worker has actually completed the save
        # (it removes the entry and adds req_id to done_sending). If we
        # cleared here, a step that never reaches execute_model would
        # silently drop the intent. See Fix B in
        # poc-runtime-issues-and-fixes.md.
        for spec in list(self._pending_saves.values()):
            meta.saves.append(spec)

        if meta.loads or meta.saves:
            logger.info(
                "[offload_poc] build_connector_meta loads=%d saves=%d "
                "pending_saves=%d done_sending=%d",
                len(meta.loads), len(meta.saves),
                len(self._pending_saves), len(self._state.done_sending))
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Record a save for this request; runs once when it finishes."""
        tokens = request.all_token_ids if hasattr(request, "all_token_ids") \
            else request.prompt_token_ids

        all_hashes = hash_prefix(tokens, self._chunk_size)
        logger.info(
            "[offload_poc] request_finished req=%s tokens=%d "
            "chunks=%d block_ids=%d",
            request.request_id, len(tokens), len(all_hashes), len(block_ids))
        if not all_hashes:
            return False, None

        # Policy: min-length gate. Skip tiny requests where the save
        # cost probably exceeds the reuse benefit.
        min_save = envs.TPU_OFFLOAD_MIN_CHUNKS_TO_SAVE
        if len(all_hashes) < min_save:
            logger.info(
                "[offload_poc] request_finished skip (below min) req=%s "
                "chunks=%d min=%d",
                request.request_id, len(all_hashes), min_save)
            return False, None

        # Policy: skip chunks that are already in the store (if enabled).
        if envs.TPU_OFFLOAD_SKIP_ALREADY_CACHED:
            new_hashes = [h for h in all_hashes if h not in self._cpu_store]
        else:
            new_hashes = list(all_hashes)

        # Policy: cap chunks saved per request.
        max_per_req = envs.TPU_OFFLOAD_MAX_CHUNKS_PER_REQ
        if len(new_hashes) > max_per_req:
            logger.info(
                "[offload_poc] request_finished cap req=%s "
                "new_chunks=%d max=%d",
                request.request_id, len(new_hashes), max_per_req)
            new_hashes = new_hashes[:max_per_req]

        if not new_hashes:
            return False, None

        # Map each new hash to the block ids backing it.
        blocks_per_chunk = self._num_blocks_for_chunks(1)
        src_block_ids: list[int] = []
        for h in new_hashes:
            idx = all_hashes.index(h)
            start = idx * blocks_per_chunk
            end = start + blocks_per_chunk
            src_block_ids.extend(block_ids[start:end])

        self._pending_saves[request.request_id] = _SaveSpec(
            req_id=request.request_id,
            chunk_hashes=new_hashes,
            src_block_ids=src_block_ids,
        )
        logger.info(
            "[offload_poc] request_finished queued_save req=%s "
            "new_chunks=%d src_blocks=%d (delay_free)",
            request.request_id, len(new_hashes), len(src_block_ids))

        # delay_free=True: ask scheduler to keep the KV blocks alive
        # until we explicitly report the request done via get_finished.
        # Worker will execute the save on an upcoming step, then add
        # req_id to done_sending.
        return True, None

    def pop_load_spec(self, req_id: str) -> _LoadSpec | None:
        """Called by the worker after load completes to release pins."""
        spec = self._pending_loads.pop(req_id, None)
        if spec is not None:
            logger.info(
                "[offload_poc] pop_load_spec req=%s released_pins=%d",
                req_id, len(spec.chunk_hashes))
            for h in spec.chunk_hashes:
                self._cpu_store.release(h)
        else:
            logger.info(
                "[offload_poc] pop_load_spec req=%s no_spec", req_id)
        return spec

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str], set[str]]:
        """Report which delay_free requests can now be freed.

        The worker pushes request ids into ``done_sending`` once their
        save has completed in ``wait_for_save``. We intersect that with
        the ids the scheduler is asking about to avoid reporting stale
        completions, and drain those that we report.

        Returns ``(done_sending, done_recving)``.
        """
        state = self._state
        ready = state.done_sending & finished_req_ids
        if ready:
            state.done_sending -= ready
            logger.info(
                "[offload_poc] get_finished report_done=%d remaining=%d",
                len(ready), len(state.done_sending))
        return ready, set()

    def _num_blocks_for_chunks(self, num_chunks: int) -> int:
        """Number of KV blocks per chunk.

        PoC assumption: the KV block size is less than or equal to the
        token chunk size, and divides it evenly. In practice the common
        config is chunk_size=256 with block_size=16 or 32.
        """
        # Deferred to runtime because vllm_config is available. Keep
        # simple: the connector only needs to know the aggregate count,
        # which the worker computes from the real runner state. For
        # the scheduler, we assume 1:1 for simplicity of this PoC
        # skeleton. A production version resolves the real ratio.
        return num_chunks


# -----------------------------------------------------------------------------
# Worker-side implementation.
# -----------------------------------------------------------------------------


class _WorkerImpl:
    """Worker-side connector logic.

    Holds the actual CPU store, performs DMA and JIT gather/scatter,
    and implements the save/load data path. The scheduler impl is
    passed in so release-on-load-complete can call back into it.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        scheduler_impl: _SchedulerImpl,
    ):
        self._state = _get_shared_state()
        self._cpu_store = self._state.cpu_store
        self._scheduler = scheduler_impl
        self._chunk_size = envs.TPU_OFFLOAD_CHUNK_SIZE
        # These are populated by register_kv_caches at engine start.
        self._mesh = None
        self._kv_cache_layout: list[jax.Array] | None = None
        self._warmed_up = False

    # ---- registration ----

    def register_kv_caches(self, kv_caches: Any) -> None:
        """Capture the KV cache handles and mesh; trigger JIT warmup.

        ``kv_caches`` is whatever vLLM passes. On tpu-inference the
        effective source of truth is the runner's ``kv_caches``
        attribute (a list of jax.Array, one per layer). For the PoC we
        accept it via this hook and also defer to the runner at
        runtime if needed.
        """
        # If vLLM hands us torch tensors, we ignore and rely on runner
        # reference later. A production connector would bridge here.
        self._kv_cache_layout = kv_caches if isinstance(kv_caches, list) \
            else None
        logger.info(
            "[offload_poc] register_kv_caches type=%s captured=%s",
            type(kv_caches).__name__, self._kv_cache_layout is not None)

    def _maybe_warmup(self, kv_caches: list[jax.Array]) -> None:
        """One-shot startup compile for the expected block count.

        Not full bucketing — just compiles gather+scatter for a
        reasonable default so the first real request does not pay the
        XLA compile cost in its TTFT. A warmup pass in the benchmark
        harness should still precede measurement.
        """
        if self._warmed_up:
            return
        if self._mesh is None and kv_caches:
            self._mesh = kv_caches[0].sharding.mesh
        if self._mesh is None:
            return
        expected_blocks = 1  # tiny warmup; real benchmark issues its own warmup
        import jax.numpy as jnp
        try:
            dummy_ids = jnp.arange(expected_blocks, dtype=jnp.int32)
            kv_caches, _ = stack_kv_cache_cross_layers(
                kv_caches, dummy_ids, expected_blocks)
            # Update the runner reference through the caller; we cannot
            # hold donated output here safely, so we just let it go out
            # of scope. The warmup is purely for the compile cache.
            del _
            self._warmed_up = True
        except Exception as exc:
            logger.warning("KV offload PoC: warmup compile failed: %s", exc)

    # ---- data path ----

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        """Load cached chunks from host DRAM back into HBM, and also
        drain any pending HBM->DRAM saves.

        Synchronous: completes before the forward pass starts.

        Saves are processed here in addition to ``wait_for_save`` because
        ``wait_for_save`` is only called after a real forward pass runs.
        When a request has just finished, the next scheduler step may
        not run ``execute_model`` and ``wait_for_save`` would be
        skipped — the save metadata would be lost. ``start_load_kv`` is
        called in every step that reaches the model runner, which is
        the safer hook in practice (matches the upstream TPUConnector
        approach).
        """
        meta = self._connector_metadata_or_none()
        runner = self._get_runner()
        logger.info(
            "[offload_poc] start_load_kv meta=%s runner=%s loads=%d "
            "saves=%d",
            meta is not None, runner is not None,
            len(meta.loads) if meta else 0,
            len(meta.saves) if meta else 0)
        if meta is None or runner is None:
            return

        # Handle saves first so the D2H copy reads the HBM blocks before
        # any concurrent free could race (delay_free protects us, but
        # being explicit is cheap).
        if meta.saves:
            self._run_saves(runner, meta)

        if not meta.loads:
            return

        # Non-jit, non-donating per-layer scatter. Each stored block
        # has shape ``(1, num_layers, block_size, num_heads, 2, head_dim)``
        # (the save path produced this via jnp.stack(axis=1) +
        # jnp.split). For every destination block we pull the layer
        # slice and do ``layer_kv = layer_kv.at[dst].set(slice)``. Done
        # outside jit so ``.sharding`` is always concrete.
        import jax.numpy as jnp
        device_sharding = jax.sharding.NamedSharding(
            self._mesh,
            runner.kv_caches[0].sharding.spec,
            memory_kind="device",
        )
        for load in meta.loads:
            logger.info(
                "[offload_poc] LOAD (DRAM->HBM) req=%s chunks=%d blocks=%d",
                load.req_id, len(load.chunk_hashes),
                len(load.dst_block_ids))
            host_chunks = [self._cpu_store.get(h) for h in load.chunk_hashes]
            if any(c is None for c in host_chunks):
                raise RuntimeError(
                    f"Offload PoC: chunk missing at load time for req "
                    f"{load.req_id}; pin-on-probe invariant violated.")

            flat_blocks: list = []
            for host_chunk in host_chunks:
                for blk in host_chunk:
                    flat_blocks.append(jax.device_put(blk, device_sharding))

            if len(flat_blocks) != len(load.dst_block_ids):
                raise RuntimeError(
                    f"Offload PoC: block count mismatch req={load.req_id} "
                    f"flat={len(flat_blocks)} dst={len(load.dst_block_ids)}")

            dst = jnp.asarray(load.dst_block_ids, dtype=jnp.int32)
            num_layers = len(runner.kv_caches)
            # Option α: push the per-layer stacking inside the jit via
            # ``_load_all`` — a single concatenate + per-layer scatter
            # in one graph. Removes the 32× outer `jnp.stack` dispatch
            # overhead that dominated the previous path.
            updated = _load_all(list(runner.kv_caches), dst, flat_blocks)
            jax.block_until_ready(updated)
            runner.kv_caches = updated
            logger.info(
                "[offload_poc] LOAD done req=%s layers_scattered=%d",
                load.req_id, num_layers)
            self._scheduler.pop_load_spec(load.req_id)

    def wait_for_save(self) -> None:
        """Save newly-filled chunks from HBM to host DRAM.

        Synchronous. Called at the forward context exit. Note: in
        practice the save usually runs via ``start_load_kv`` (see the
        comment there). This hook is kept as a fallback for scheduler
        steps that do reach forward-pass completion.
        """
        meta = self._connector_metadata_or_none()
        runner = self._get_runner()
        logger.info(
            "[offload_poc] wait_for_save meta=%s runner=%s saves=%d",
            meta is not None, runner is not None,
            len(meta.saves) if meta else 0)
        if meta is None or runner is None or not meta.saves:
            return
        self._run_saves(runner, meta)

    def _run_saves(self, runner, meta: "_PoCConnectorMetadata") -> None:
        """Do the HBM->DRAM copy for any saves in ``meta``.

        Uses a non-donating gather so the runner's ``kv_caches`` arrays
        are preserved intact for the upcoming forward pass. This avoids
        the "Array has been deleted" error we hit when we first tried
        ``stack_kv_cache_cross_layers`` (which donates kv_caches).

        Idempotent: entries already drained from ``pending_saves`` are
        skipped. Removes the entry from ``pending_saves`` and adds the
        req_id to ``done_sending`` on success.
        """
        import jax.numpy as jnp
        if self._mesh is None and runner.kv_caches:
            self._mesh = runner.kv_caches[0].sharding.mesh

        host_sharding = jax.sharding.NamedSharding(
            self._mesh,
            runner.kv_caches[0].sharding.spec,
            memory_kind="pinned_host",
        )
        device_sharding_spec = runner.kv_caches[0].sharding.spec

        for save in meta.saves:
            if save.req_id not in self._state.pending_saves:
                continue

            block_ids = save.src_block_ids
            logger.info(
                "[offload_poc] SAVE (HBM->DRAM) req=%s chunks=%d blocks=%d",
                save.req_id, len(save.chunk_hashes), len(block_ids))

            # Non-donating gather: read each layer's slice for the
            # requested block ids, without invalidating runner.kv_caches.
            block_ids_arr = jnp.asarray(block_ids, dtype=jnp.int32)
            gathered_per_layer = [
                layer_kv.at[block_ids_arr].get()
                for layer_kv in runner.kv_caches
            ]
            # stacked shape: (num_blocks, num_layers, ...) via stack on axis=1
            stacked = jnp.stack(gathered_per_layer, axis=1)
            per_block = jnp.split(
                stacked, indices_or_sections=len(block_ids), axis=0)

            chunks_per_chunk_hash = len(block_ids) // len(save.chunk_hashes) \
                if save.chunk_hashes else 1
            for i, chunk_hash in enumerate(save.chunk_hashes):
                start = i * chunks_per_chunk_hash
                end = start + chunks_per_chunk_hash
                blocks_for_chunk = per_block[start:end]
                host_copies = []
                for blk in blocks_for_chunk:
                    host_dest = jax.device_put(
                        jax.numpy.zeros_like(blk), host_sharding)
                    host_blk = kv_transfer.copy_to_host(
                        blk, host_dest, self._mesh, device_sharding_spec)
                    host_copies.append(host_blk)
                jax.block_until_ready(host_copies)
                logger.info(
                    "[offload_poc] SAVE chunk_hash=%s blocks_done=%d",
                    str(chunk_hash)[:16], len(host_copies))
                self._cpu_store.put(chunk_hash, host_copies)

            self._state.pending_saves.pop(save.req_id, None)
            self._state.done_sending.add(save.req_id)
            logger.info(
                "[offload_poc] SAVE done req=%s done_sending=%d",
                save.req_id, len(self._state.done_sending))

    def _connector_metadata_or_none(self) -> _PoCConnectorMetadata | None:
        # Provided by KVConnectorBase_V1; the outer facade exposes it
        # via _get_connector_metadata. We look it up lazily.
        return getattr(self, "_meta_ref", None)

    def _get_runner(self):
        # The facade wires this up in register_runner (see below).
        return getattr(self, "_runner", None)


# -----------------------------------------------------------------------------
# Public connector facade.
# -----------------------------------------------------------------------------


class TPUOffloadConnector(KVConnectorBase_V1):
    """KV cache offload connector (HBM <-> host DRAM).

    This facade routes KVConnectorBase_V1 calls to the scheduler or
    worker implementation based on ``role``.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._role = role

        # Shared state lives in the module-level singleton (see
        # _SharedState above) so scheduler- and worker-side connector
        # instances see the same DRAM store and the same pending-save /
        # pending-load dicts.
        self._scheduler_impl = _SchedulerImpl(vllm_config)
        if role == KVConnectorRole.WORKER:
            self._worker_impl = _WorkerImpl(
                vllm_config, self._scheduler_impl)
        else:
            self._worker_impl = None
        logger.info(
            "[offload_poc] TPUOffloadConnector __init__ role=%s "
            "chunk_size=%d num_cpu_chunks=%d shared_store_len=%d",
            role, envs.TPU_OFFLOAD_CHUNK_SIZE,
            envs.TPU_OFFLOAD_NUM_CPU_CHUNKS,
            len(_get_shared_state().cpu_store))

    # ---- convenience for integration ----

    def register_runner(self, runner) -> None:
        """Called from tpu_inference to give us a handle to the runner.

        Not part of the base KVConnectorBase_V1 API; tpu-inference
        invokes this after constructing the runner so we can read its
        ``kv_caches`` at load/save time.
        """
        logger.info(
            "[offload_poc] register_runner called on role=%s worker_impl=%s",
            getattr(self, "_role", "?"), self._worker_impl is not None)
        if self._worker_impl is not None:
            self._worker_impl._runner = runner

    # ==============================
    # Scheduler-side routes
    # ==============================

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self._scheduler_impl.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return self._scheduler_impl.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        return self._scheduler_impl.build_connector_meta(scheduler_output)

    def request_finished(self, request, block_ids):
        return self._scheduler_impl.request_finished(request, block_ids)

    def get_finished(self, finished_req_ids):
        return self._scheduler_impl.get_finished(finished_req_ids)

    # ==============================
    # Worker-side routes
    # ==============================

    def register_kv_caches(self, kv_caches) -> None:
        if self._worker_impl is not None:
            self._worker_impl.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self._worker_impl is not None:
            self._worker_impl._meta_ref = connector_metadata

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self._worker_impl is not None:
            self._worker_impl._meta_ref = None

    def start_load_kv(self, forward_context, **kwargs) -> None:
        if self._worker_impl is not None:
            self._worker_impl.start_load_kv(forward_context, **kwargs)

    def wait_for_save(self) -> None:
        if self._worker_impl is not None:
            self._worker_impl.wait_for_save()

    # ---- abstract layer-wise methods: no-op (we save/load in bulk) ----

    def wait_for_layer_load(self, layer_name: str) -> None:
        # PoC does bulk load in start_load_kv; per-layer waits are no-ops.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # PoC does bulk save in wait_for_save; per-layer saves are no-ops.
        return


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _flatten_block_ids(blocks: Any) -> list[int]:
    """Best-effort flatten of vLLM's KVCacheBlocks into a list of ints.

    The KVCacheBlocks object can take several shapes across vLLM
    versions (including multi-level nesting for HMA/group layouts).
    Recurse into nested containers.
    """

    def _walk(obj: Any, out: list[int]) -> None:
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for x in obj:
                _walk(x, out)
            return
        # Scalar: try to coerce to int.
        try:
            out.append(int(obj))
        except (TypeError, ValueError):
            return

    if blocks is None:
        return []
    flat: list[int] = []
    if hasattr(blocks, "block_ids"):
        _walk(blocks.block_ids, flat)
        if flat:
            return flat
    if hasattr(blocks, "get_block_ids"):
        _walk(list(blocks.get_block_ids()), flat)
        return flat
    try:
        _walk(list(blocks), flat)
    except TypeError:
        pass
    return flat
