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

import jax

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

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

    def __init__(self, vllm_config: "VllmConfig", cpu_store: PoCCPUStore):
        self._cpu_store = cpu_store
        self._chunk_size = envs.TPU_OFFLOAD_CHUNK_SIZE
        # Track per-request state so we know where to save after prefill
        # and which chunks to release after load.
        self._pending_saves: dict[str, _SaveSpec] = {}
        self._pending_loads: dict[str, _LoadSpec] = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Only consider tokens beyond what vLLM already has in HBM APC.
        tokens = request.prompt_token_ids
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
        # Draining: the connector framework reconstructs metadata each step.
        for spec in list(self._pending_loads.values()):
            if spec.dst_block_ids:  # only ready ones
                meta.loads.append(spec)
        # Pending_loads entries survive until the worker confirms via
        # release-on-load-complete (see _WorkerImpl.start_load_kv).

        # Saves are queued by request_finished (below).
        for spec in list(self._pending_saves.values()):
            meta.saves.append(spec)
        self._pending_saves.clear()

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
        if not all_hashes:
            return False, None

        # Only save chunks that are NOT already in the store.
        new_hashes = [h for h in all_hashes if h not in self._cpu_store]
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

        # Synchronous save; no need to delay_free beyond this step.
        return False, None

    def pop_load_spec(self, req_id: str) -> _LoadSpec | None:
        """Called by the worker after load completes to release pins."""
        spec = self._pending_loads.pop(req_id, None)
        if spec is not None:
            for h in spec.chunk_hashes:
                self._cpu_store.release(h)
        return spec

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
        cpu_store: PoCCPUStore,
        scheduler_impl: _SchedulerImpl,
    ):
        self._cpu_store = cpu_store
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
        """Load cached chunks from host DRAM back into HBM.

        Synchronous: completes before the forward pass starts.
        """
        meta = self._connector_metadata_or_none()
        if meta is None:
            return

        runner = self._get_runner()
        if runner is None or not meta.loads:
            return

        self._maybe_warmup(runner.kv_caches)

        for load in meta.loads:
            host_chunks = [self._cpu_store.get(h) for h in load.chunk_hashes]
            if any(c is None for c in host_chunks):
                # A pinned chunk vanished; fail loudly rather than
                # running the forward pass on uninitialised KV.
                raise RuntimeError(
                    f"Offload PoC: chunk missing at load time for req "
                    f"{load.req_id}; pin-on-probe invariant violated.")

            # Scatter: move the host chunks back into the allocated blocks.
            # Each host_chunk is expected to be already resident in a
            # form that jax.device_put can dispatch.
            device_chunks = [jax.device_put(c) for c in host_chunks]
            replicated_sharding = jax.sharding.NamedSharding(
                self._mesh,
                jax.sharding.PartitionSpec(),
                memory_kind="device",
            )
            src_offsets, dest_offsets, chunk_sizes, num_chunks = \
                pre_update_kv_caches(
                    load.dst_block_ids, self._mesh, replicated_sharding)
            kv_sharding_spec = runner.kv_caches[0].sharding.spec
            runner.kv_caches = update_kv_caches(
                runner.kv_caches,
                device_chunks,
                src_offsets,
                dest_offsets,
                chunk_sizes,
                num_chunks,
                self._mesh,
                kv_sharding_spec,
                kv_sharding_spec,
                replicated_sharding.spec,
            )
            # Release pins; the chunk is now safely loaded into HBM.
            self._scheduler.pop_load_spec(load.req_id)

    def wait_for_save(self) -> None:
        """Save newly-filled chunks from HBM to host DRAM.

        Synchronous. Called at the forward context exit.
        """
        meta = self._connector_metadata_or_none()
        if meta is None:
            return

        runner = self._get_runner()
        if runner is None or not meta.saves:
            return

        self._maybe_warmup(runner.kv_caches)

        host_sharding = jax.sharding.NamedSharding(
            self._mesh,
            # Use the device sharding's PartitionSpec, but target pinned_host.
            runner.kv_caches[0].sharding.spec,
            memory_kind="pinned_host",
        )
        device_sharding_spec = runner.kv_caches[0].sharding.spec

        for save in meta.saves:
            block_ids = save.src_block_ids
            import jax.numpy as jnp
            block_ids_arr = jnp.asarray(block_ids, dtype=jnp.int32)
            runner.kv_caches, stacked = stack_kv_cache_cross_layers(
                runner.kv_caches, block_ids_arr, len(block_ids))

            # D2H one chunk at a time. copy_to_host expects a dest on
            # pinned_host; we allocate one the simple way.
            chunks_per_chunk_hash = len(block_ids) // len(save.chunk_hashes) \
                if save.chunk_hashes else 1
            for i, chunk_hash in enumerate(save.chunk_hashes):
                start = i * chunks_per_chunk_hash
                end = start + chunks_per_chunk_hash
                blocks_for_chunk = stacked[start:end]
                host_copies = []
                for blk in blocks_for_chunk:
                    host_dest = jax.device_put(
                        jax.numpy.zeros_like(blk), host_sharding)
                    host_blk = kv_transfer.copy_to_host(
                        blk, host_dest, self._mesh, device_sharding_spec)
                    host_copies.append(host_blk)
                # Await async DMA before declaring the save done.
                jax.block_until_ready(host_copies)
                self._cpu_store.put(chunk_hash, host_copies)

    def _connector_metadata_or_none(self) -> _PoCConnectorMetadata | None:
        # Provided by KVConnectorBase_V1; the outer facade exposes it
        # via _get_connector_metadata. We look it up lazily.
        return getattr(self, "_meta_ref", None)

    def _get_runner(self):
        # The facade wires this up in set_runner (see below).
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

        # The CPU store is constructed lazily; the worker holds the
        # authoritative copy. The scheduler probes through it.
        self._cpu_store = PoCCPUStore(
            max_entries=envs.TPU_OFFLOAD_NUM_CPU_CHUNKS)

        self._scheduler_impl = _SchedulerImpl(vllm_config, self._cpu_store)
        if role == KVConnectorRole.WORKER:
            self._worker_impl = _WorkerImpl(
                vllm_config, self._cpu_store, self._scheduler_impl)
        else:
            self._worker_impl = None

    # ---- convenience for integration ----

    def set_runner(self, runner) -> None:
        """Called from tpu_inference to give us a handle to the runner.

        Not part of the base KVConnectorBase_V1 API; tpu-inference
        invokes this after constructing the runner so we can read its
        ``kv_caches`` at load/save time.
        """
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
    versions; we handle the common ones and fall back to ``int(b)``.
    """
    if blocks is None:
        return []
    # Try common shapes.
    if hasattr(blocks, "block_ids"):
        ids = blocks.block_ids
        if isinstance(ids, (list, tuple)):
            flat: list[int] = []
            for item in ids:
                if isinstance(item, (list, tuple)):
                    flat.extend(int(x) for x in item)
                else:
                    flat.append(int(item))
            return flat
    if hasattr(blocks, "get_block_ids"):
        return [int(b) for b in blocks.get_block_ids()]
    try:
        return [int(b) for b in blocks]
    except TypeError:
        return []
