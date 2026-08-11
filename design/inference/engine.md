---
applies_to:
  - inference/crates/icn-engine/**
  - inference/crates/icn-mtp/**
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
  `-- optional MTP target/draft state
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
- Assessment and loading use the same MTP selector and policy fingerprint.
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
  -> retain committed text prefix when safe
  -> terminal result
```

Prompt progress becomes committed only after target decode and linked MTP processing succeed.
Batch effects are staged separately from requests until that boundary. A sampled token becomes
committed only when a later decode or speculative-verification step accepts it.

Lifecycle observations report queueing, preparation, prefill, and generation start. They are
coalesced, rate-limited latest-state signals and may be replaced rather than delay inference.
Semantic output and terminal results retain bounded backpressure.

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
chooses the longest exact token match and transfers the sequence into active ownership. Native KV
never leaves llama.cpp.

Cancellation and stream disconnection do not make committed text state ambiguous. They preserve it
when cache policy permits. See [KV state reuse](./kv.md) for invalidation rules.

## Failure and shutdown

| Event | Effect |
| --- | --- |
| Validation, cancellation, disconnection, request-local failure | Affect one request; retain its committed text prefix when eligible |
| Shared native batch failure | Reset target/draft context state and invalidate all prefixes in that context |
| Sequence cleanup failure | Quarantine that sequence ID |
| Shutdown | Reject queued work, fail active work, release state, join the executor |

Prompt reuse is optional: missing state always falls back to cold prefill.

## Output and observability

Token output passes through UTF-8 buffering, stop detection, and semantic parsing. Transport-specific
tool-call policy remains outside the native parser.

Final metrics cover queue, prompt, decode, first-token latency, throughput, sampler/parser time,
reused prompt tokens, and MTP draft/acceptance/verification when enabled. The server also reports the
resolved execution configuration. Lifecycle control chunks contain no semantic choices.

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
