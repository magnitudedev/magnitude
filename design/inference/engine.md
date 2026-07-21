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

The Magnitude inference engine is a persistent, per-model serving runtime built around pinned upstream llama.cpp. It converts typed chat requests into token work, continuously batches multiple sequences, streams semantic output, and retains reusable prompt state. The engine is deliberately opinionated: one thread owns native model state, while bounded channels isolate callers and transport backpressure from that state.

Two subsystem documents carry the detailed policies:

- [KV state reuse](./kv.md) describes standard llama.cpp sequence-state reuse and its safety boundary.
- [Scheduler design](./scheduler.md) describes admission, batching, request state transitions, fairness, and failure handling.

## Influences

The engine keeps **llama.cpp** as its native model runtime while Magnitude owns request admission,
batching, streaming, and sequence assignment. Prompt reuse stays within llama.cpp's standard sequence
state primitives; Magnitude does not duplicate the native KV representation.

## System shape

```text
API / caller threads
        │ bounded commands
        ▼
per-model executor thread
  ├── chat preparation and tokenization
  ├── waiting queue and active request state
  ├── scheduler and sequence pool
  │      └── retained llama.cpp sequence state
  ├── llama.cpp target context
  ├── optional multimodal runtime
  └── optional MTP target/draft operations
        │ bounded result streams
        ▼
API / caller threads
```

`LlamaCompletionBackend` is the public handle. Loading it accepts serializable execution intent,
starts a named executor thread, initializes the native backend, creates a fresh process-local backend
plan, and consumes that exact owned plan to initialize the model, context, chat templates, worker
pools, and optional projector or MTP runtime. It then returns a typed readiness result which
distinguishes invalid or incompatible artifacts, `DoesNotFit`, operational planning failure,
allocation failure, and success with normalized resolved evidence. The
handle exposes completion, template application, model properties, and idle-only native planning
operations.

Execution intent is policy, not an executable plan. Native defaults, tensor splits, arbitrary
tensor-buffer placements, model/context parameters, and auxiliary context parameters are built in
one ICN planner implementation. They are retained in pointer-safe owned native objects and passed
directly to loading. The engine never reconstructs them from a serialized summary. Preview uses the
same planner but destroys the process-local plan and caches only normalized assessment evidence;
loading always replans under current conditions.

## Ownership and concurrency

One executor thread exclusively owns each model's mutable native resources. This gives the engine a clear serialization boundary for llama.cpp memory operations, sequence mutations, sampling state, and shutdown. Callers may be concurrent, but they communicate with the owner rather than locking the native context directly.

The executor also exclusively owns process-global backend initialization for a resident runtime.
Load-time MTP selection, fit planning, and model construction occur in that same initialized backend
session. The selected MTP configuration remains part of the execution intent passed to fitting, so
target and draft memory are assessed together. A serving-process preflight must never initialize a
temporary backend before invoking the executor. Model-free preview and inventory assessment use
isolated worker processes so their backend lifetime cannot conflict with the resident executor.
The same MTP selector implementation and policy fingerprint are used in isolated assessment and
resident loading. Target identity includes the selected component set and serving-configuration
revision, and parity tests compare normalized fit evidence with loaded execution evidence.

There are three bounded flows:

- the model command queue bounds queued demand;
- each request's native-to-caller event channel bounds transport buffering;
- each active request's small outbound queue decouples native scheduling from a briefly slow consumer.

Read-only hardware observations share the bounded model command channel. They run between scheduler
batches and may inspect backend device memory plus
immutable generation allocation evidence. They cannot mutate model/context state, plan a load, or
delay until all inference becomes idle. General native planning callbacks remain idle-only.

This design avoids unbounded queues and makes overload explicit. It also means synchronous native work and current KV tier I/O can pause progress for all sequences owned by that executor.

## Request lifecycle

At a high level, a completion follows this path:

1. The caller submits a typed chat request and cancellation flag.
2. The executor validates/prepares the chat template and queues the request.
3. The scheduler tokenizes and admits it when a sequence and KV capacity are available.
4. A reusable prompt prefix is retained or restored from standard sequence state when eligible.
5. Prefill and decode tokens join continuous native batches.
6. Ready logits are sampled and decoded into UTF-8 and semantic stream events.
7. Stop conditions, generation limits, cancellation, or errors make the request terminal.
8. Committed KV history is retained when eligible, the sequence is cleared/released, and a final generation or failure is delivered.

Prompt K/V is considered committed only after native decode succeeds. The currently sampled token remains outside committed history until a subsequent decode or verification step commits it. That boundary protects both cache correctness and recovery after speculative or native failure.

## Prompt state

The engine retains committed prompt state per free native sequence and assigns the longest exact
prefix match at admission. This avoids a second physical-cache abstraction and works for every
architecture supported by the selected upstream llama.cpp primitives. Unified KV remains an
ordinary native execution option, not a prerequisite for a Magnitude-specific page cache.

## Scheduler loop

Each executor iteration performs a bounded amount of orchestration:

1. Drain new commands.
2. Service one read-only hardware observation when pending.
3. Run exclusive native tasks only if inference is idle.
4. Clean up terminal or disconnected requests.
5. Admit queued completions while sequences and KV capacity permit.
6. Sample requests whose logits are ready and update committed sequence histories.
7. Build and execute one decode/prefill batch.
8. Flush outputs and clean up again.
9. Poll for commands briefly when no native work ran.

The detailed admission and batching policy is in [scheduler.md](./scheduler.md). The important architectural property is that scheduling decisions and the KV mutations they depend on occur under the same single-owner loop.

## Output and observability

Native token results pass through UTF-8 buffering, stop detection, and a semantic stream parser before reaching the API. Transport-specific tool-call policy remains outside the native parser. Timing snapshots can be emitted with stream events, and final generation metrics include:

- queue, prompt, decode, and time-to-first-token durations;
- prompt and decode throughput;
- sampler and parser time;
- reused prompt-token counts;
- speculative draft, acceptance, and verification metrics when MTP is active.

The server reports the effective resolved execution configuration so observed behavior can be tied to the loaded plan.

## Failure model

Prompt reuse is an optimization; unavailable state falls back to cold prefill. Request-local validation and cancellation fail only the affected request where possible. Native batch failure is treated more conservatively because several sequences may share one context and a single decode call: the executor synchronizes and clears affected native memory before admitting more work.

Sequence IDs whose cleanup fails are quarantined rather than returned to the free pool. Shutdown rejects new work, fails queued work, releases active state, and joins the owner thread.

## Design boundaries

The current engine is not a distributed cache or a general GPU serving fabric. It has no
cross-process KV sharing, restart-persistent KV state, request migration, paged-attention block
table, prefix-aware queue ordering, or benefit-aware preemption. Any future low-level native change
must establish that upstream primitives cannot provide the required behavior and requires explicit
approval under the inference fork-maintenance policy.
