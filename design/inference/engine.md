---
applies_to:
  - inference/crates/icn-engine/**
  - inference/crates/icn-mtp/**
  - inference/crates/icn-contracts/src/lib.rs
  - inference/crates/icn-contracts/src/output.rs
  - inference/crates/icn-api/src/lib.rs
  - inference/crates/icn-server/src/main.rs
  - inference/native/llama-cpp-rs/llama-cpp-2/**
---

# Inference engine design

The Magnitude inference engine is a persistent, per-model serving runtime built around a pinned llama.cpp fork. It converts typed chat requests into token work, continuously batches multiple sequences, streams semantic output, and retains reusable prompt state. The engine is deliberately opinionated: one thread owns native model state, while bounded channels isolate callers and transport backpressure from that state.

Two subsystem documents carry the detailed policies:

- [KV cache design](./kv.md) describes radix-prefix identity, native page ownership, RAM/disk tiers, and eviction safety.
- [Scheduler design](./scheduler.md) describes admission, batching, request state transitions, fairness, and failure handling.

## Influences

The engine keeps **llama.cpp** as its native model runtime but replaces llama-server's externally visible slot/cache policy with Magnitude-owned request state, scheduling, and prefix reuse. **SGLang** is the main influence for dynamic radix-prefix sharing and cache-aware concurrency. **vLLM** is the main influence for page-granular KV ownership, strict allocation accounting, and scheduling work according to immediately required cache capacity. The result is a llama.cpp-backed engine with serving policies adapted from both systems, not a compatibility clone of any of them.

## System shape

```text
API / caller threads
        │ bounded commands
        ▼
per-model executor thread
  ├── chat preparation and tokenization
  ├── waiting queue and active request state
  ├── scheduler and sequence pool
  │      └── radix KV cache ── device / RAM / disk
  ├── llama.cpp target context
  ├── optional multimodal runtime
  └── optional MTP target/draft operations
        │ bounded result streams
        ▼
API / caller threads
```

`LlamaCompletionBackend` is the public handle. Loading it validates a fully resolved execution plan, starts a named executor thread, initializes the backend, model, context, chat templates, worker pools, and optional projector or MTP runtime, and waits for an explicit readiness result. The handle exposes completion, template application, model properties, and idle-only native planning operations.

The resolved execution plan is the shared contract between hardware assessment, server status, and loading. The engine does not independently invent context, batching, KV, offload, or worker defaults after assessment.

## Ownership and concurrency

One executor thread exclusively owns each model's mutable native resources. This gives the engine a clear serialization boundary for llama.cpp memory operations, radix mutations, sampling state, and shutdown. Callers may be concurrent, but they communicate with the owner rather than locking the native context directly.

There are three bounded flows:

- the model command queue bounds queued demand;
- each request's native-to-caller event channel bounds transport buffering;
- each active request's small outbound queue decouples native scheduling from a briefly slow consumer.

This design avoids unbounded queues and makes overload explicit. It also means synchronous native work and current KV tier I/O can pause progress for all sequences owned by that executor.

## Request lifecycle

At a high level, a completion follows this path:

1. The caller submits a typed chat request and cancellation flag.
2. The executor validates/prepares the chat template and queues the request.
3. The scheduler tokenizes and admits it when a sequence and KV capacity are available.
4. A reusable prompt prefix is attached from the radix cache or restored from the legacy sequence cache when eligible.
5. Prefill and decode tokens join continuous native batches.
6. Ready logits are sampled and decoded into UTF-8 and semantic stream events.
7. Stop conditions, generation limits, cancellation, or errors make the request terminal.
8. Committed KV history is retained when eligible, the sequence is cleared/released, and a final generation or failure is delivered.

Prompt K/V is considered committed only after native decode succeeds. The currently sampled token remains outside committed history until a subsequent decode or verification step commits it. That boundary protects both cache correctness and recovery after speculative or native failure.

## Serving modes

The engine has two prompt-cache modes selected at runtime:

| Mode | When used | Reuse semantics |
| --- | --- | --- |
| Radix pages | Enabled, no MTP/draft context, and native unified pages supported | Dynamic prefix reuse across requests with shared physical K/V pages and optional RAM/disk spill |
| Legacy sequence cache | Radix unavailable or disabled | Retained per-sequence prompt checkpoints; a free sequence is selected by longest matching prefix |

Radix mode is text-only today. Multimodal requests and native memory layouts whose position/state semantics are not safely pageable bypass it. MTP maintains linked target and draft state and therefore also uses the non-radix path.

Unified KV is required for physical page sharing and lower tiers, but is not the global default. In llama.cpp's current dense attention kernel, unified KV can improve shared-prefix concurrency while hurting unrelated concurrent work. The engine exposes the choice rather than silently imposing it.

## Scheduler loop

Each executor iteration performs a bounded amount of orchestration:

1. Drain new commands.
2. Run exclusive native tasks only if inference is idle.
3. Clean up terminal or disconnected requests.
4. Admit queued completions while sequences and KV capacity permit.
5. Sample requests whose logits are ready and publish committed prefix pages.
6. Build and execute one decode/prefill batch.
7. Flush outputs and clean up again.
8. Poll for commands briefly when no native work ran.

The detailed admission and batching policy is in [scheduler.md](./scheduler.md). The important architectural property is that scheduling decisions and the KV mutations they depend on occur under the same single-owner loop.

## Output and observability

Native token results pass through UTF-8 buffering, stop detection, and a semantic stream parser before reaching the API. Transport-specific tool-call policy remains outside the native parser. Timing snapshots can be emitted with stream events, and final generation metrics include:

- queue, prompt, decode, and time-to-first-token durations;
- prompt and decode throughput;
- sampler and parser time;
- cached tokens sourced from device, RAM, and disk;
- lower-tier promotion time;
- speculative draft, acceptance, and verification metrics when MTP is active.

The server reports the effective resolved execution configuration, including radix cache settings, so observed behavior can be tied to the loaded plan.

## Failure model

The cache is an optimization; invalid spill data falls back to cold prefill. Request-local validation and cancellation fail only the affected request where possible. Native batch failure is treated more conservatively because several sequences may share one context and a single decode call: the executor synchronizes and clears affected native memory before admitting more work.

Sequence IDs whose cleanup fails are quarantined rather than returned to the free pool. Shutdown rejects new work, fails queued work, releases active state, removes process-scoped spill storage, and joins the owner thread.

## Design boundaries

The current engine is not a distributed cache or a general GPU serving fabric. It has no cross-process KV sharing, restart-persistent spill index, async KV transfer pipeline, request migration, or paged-attention block table. It also does not yet implement SGLang/vLLM-level prefix-aware queue ordering or benefit-aware preemption.

The existing architecture establishes the correctness boundary needed for those improvements: explicit native page ownership, bounded tier accounting, a cache-aware admission protocol, a single serialized mutation owner, and metrics that distinguish saved compute from transfer work.
