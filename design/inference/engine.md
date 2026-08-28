---
applies_to:
  - inference/crates/icn-engine/**
  - inference/crates/icn-speculative/**
  - inference/crates/icn-contracts/src/lib.rs
  - inference/crates/icn-contracts/src/output.rs
  - inference/crates/icn-api/src/lib.rs
  - inference/crates/icn-server/src/main.rs
  - inference/crates/icn-server/src/inference_worker.rs
  - inference/crates/icn-server/src/memory_supervisor.rs
  - inference/native/llama-cpp-rs/llama-cpp-2/**
---

# Inference engine

The inference engine is a persistent, per-model llama.cpp runtime. Magnitude owns request
admission, batching, streaming, and sequence assignment; llama.cpp owns model execution and native
KV state.

Detailed policy lives in:

- [KV state reuse](./kv.md)
- [Scheduler](./scheduler.md)
- [System memory management](./system-memory-management.md)

## Runtime shape

```text
persistent ICN
  |
  | bounded IPC
  v
disposable inference worker (one resident model)
  |
  | bounded commands
  v
single executor thread
  +-- request preparation
  +-- scheduler + sequence pool
  +-- llama.cpp target context
  +-- optional projector
  `-- optional speculative target/draft state
  |
  | bounded result streams
  v
caller
```

The executor thread is the only owner and mutator of resident native state. Callers are concurrent;
native model, context, sequence, sampler, and shutdown operations are serialized.

Persistent ICN owns no resident executor. Worker exit reclaims the complete resident topology.

## Planning and loading

Execution intent is policy, not a native plan. One planner resolves native defaults, device and
tensor placement, model/context parameters, and optional components.

```text
execution intent
      |
      v
native planner -----> normalized assessment evidence
      |
      `-------------> process-local owned plan -----> resident load
```

Rules:

- Native plans are pointer-safe, process-local values and are never reconstructed from summaries.
- Assessment discards its plan; resident loading replans under current conditions.
- Assessment and loading use the same speculative-decoding selector and policy fingerprint.
- Execution identity includes the selected components and serving-configuration revision.
- A serving worker initializes one installation-authorized native backend for its lifetime.
- Load success requires exhaustive resident-allocation evidence with every location resolved to one
  physical memory domain.
- Load readiness distinguishes incompatible artifacts, insufficient memory, planning failure,
  allocation failure, and success with normalized evidence.

A resolved sequence capacity of `P` provisions `configured context x P` physical context while each
request remains capped at the configured context. The current allocator uses partitioned native KV
to preserve that per-request limit.

Prepared execution reports completed semantic phases—target model, context, optional components,
setup, warm-up, and finalization. These phases are not byte-level native progress.

## Request flow

```text
submit
  -> prepare + tokenize
  -> wait for sequence capacity
  -> restore exact reusable prefix or cold-prefill
  -> prefill / decode / sample
  -> stop, cancel, disconnect, or fail
  -> retain committed semantic prefix when safe
  -> terminal usage, termination, and metrics facts
```

Prompt progress becomes committed only after target decode and linked speculative processing
succeed, including every token or embedding sub-batch in a multimodal prompt.
Batch effects are staged separately from requests until that boundary. A sampled token becomes
committed only when a later decode or speculative-verification step accepts it.
Generation is always bounded by the request's remaining configured context capacity. A caller may
add a smaller output-token limit; when it does not, remaining context capacity is the sole limit.
Multimodal projector execution supports only speculative methods that can advance from embedding
sub-batches; MTP is rejected during configuration validation.

Linked speculative execution carries two explicit coordinates at every boundary: the target's
native position and the draft context's consecutive logical-token position. Text-only execution
advances them together. M-RoPE media may advance the target coordinate differently, so the binding
mirrors target batch rows through a lightweight position view rather than passing target positions
to the draft model. Draft preparation, verification, rollback, and trimming consume the same
linked boundary; there is no independently inferred draft cursor.

Prepared prompts use one semantic representation for text-only and multimodal requests. Text spans
contain exact model tokens; media spans contain a stable content identity, their logical
token cost, and their native position cost. Exact media spans can therefore reuse resident KV just
like text while remaining indivisible. The binding exposes upstream single-chunk MTMD evaluation;
Magnitude does not modify or recreate llama.cpp projector behavior.

Lifecycle observations report queueing, preparation, prefill, and generation start. They are
coalesced, rate-limited latest-state signals and may be replaced rather than delay inference.
Semantic output and terminal facts retain bounded backpressure. The engine never rebuilds semantic
output from its own events; the caller-owned output journal performs that aggregation once.

## Bounded concurrency

| Flow | Bound |
| --- | --- |
| Host to worker | Framed IPC queues |
| Worker to executor | Model command queue |
| Executor to request | Per-request event channel |
| Native scheduling to transport | Small per-request outbound queue |

A slow consumer pauses native work only for its request. Synchronous native operations may still
pause all sequences on the executor.

Read-only hardware observations share the command queue and run between batches. They may inspect
device memory and immutable allocation evidence but cannot mutate state or plan a load. Other native
planning work is idle-only.

## Prompt state

An available native sequence carries its optional reusable prefix as one owned value. Admission
chooses the longest exact semantic match and transfers the sequence into active ownership. Native
KV never leaves llama.cpp. Logical token counts drive capacity and progress; native positions drive
KV trimming and continuation, including M-RoPE prompts where those values differ.
When speculation is active, a reusable checkpoint is one value containing the linked boundary and
a binding-owned prompt state with target state, draft state, and any method-owned state. A
target-only or partially restored speculative checkpoint is not representable.

Cancellation and stream disconnection do not make committed semantic state ambiguous. They
preserve it when cache policy permits. See [KV state reuse](./kv.md) for invalidation rules.

## Failure and shutdown

| Event | Effect |
| --- | --- |
| Validation, cancellation, disconnection, request-local failure | Affect one request; retain its committed semantic prefix when eligible |
| Shared native batch failure | Reset target/draft context state and invalidate all prefixes in that context |
| Sequence cleanup failure | Quarantine that sequence ID |
| Shutdown | Reject queued work, fail active work, release state, join the executor |

Prompt reuse is optional: missing state always falls back to cold prefill.

## Output and observability

Token output passes through UTF-8 buffering, stop detection, and semantic parsing. Transport-specific
tool-call policy remains outside the native parser.

Final metrics cover queue, prompt, decode, first-token latency, throughput, sampler/parser time,
reused prompt tokens, and speculative draft/acceptance/verification when enabled. The server also
reports the selected method, effective draft bounds, and resolved execution configuration.
Lifecycle control chunks contain no semantic choices.

## Boundaries

The engine provides no cross-process or restart-persistent KV sharing, request migration,
paged-attention block table, prefix-aware queue ordering, or benefit-aware preemption. A new native
fork must first establish that upstream primitives cannot provide the required behavior and receive
explicit approval under the inference fork-maintenance policy.

## Acceptance criteria

- One executor exclusively owns all mutable native state for a loaded model.
- All command and result paths are bounded.
- Request-visible prompt progress reflects committed native work only.
- Prompt reuse is optional and cannot change inference semantics.
- Assessment and resident loading produce comparable normalized execution evidence.
- MTP, DFlash, and DSpark use one shared speculative lifecycle; method-specific behavior remains
  explicit and is delegated to the pinned native implementation.
- Target and draft state advance atomically. Failure after either side advances resets both before
  the sequence can be reused.
- Every speculative operation uses an explicit target-native/draft-sequential position pair.
