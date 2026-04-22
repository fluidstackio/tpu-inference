# SPDX-License-Identifier: Apache-2.0
"""Minimal KV cache offload PoC connector.

This package implements a feasibility-grade HBM <-> host DRAM KV cache
offload connector as proposed in the RFC `rfc-ext-kv-cache-mvp-poc.md`.

Scope is deliberately narrow:
  - DRAM only (no NVMe, no remote, no RDMA).
  - Single host, single instance.
  - Opt-in via vLLM's --kv-transfer-config; default off.

The package is named ``offload_poc`` rather than ``offload`` to avoid
colliding with future production code, and to signal that the
implementation trades breadth for minimal surface area suitable for
answering one question: does DRAM-level KV offload produce measurable
TTFT improvement on TPU for workloads with long shared prefixes?
"""
