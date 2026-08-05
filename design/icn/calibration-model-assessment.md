---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-server/**
  - inference/crates/icn-api/**
  - inference/native/llama-cpp-rs/**
  - packages/icn-protocol/**
  - packages/acn/src/local-model*.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - cli/src/features/model-setup/**
  - cli/src/features/model-menus/**
---

# Hardware calibration and model assessment

## Ownership

| Concern | Owner |
|---|---|
| Hardware discovery, calibration, native planning, memory and performance evidence | ICN |
| Profile policy, assessment orchestration, recommendations | ACN |
| Presentation | Clients |

## Terms

| Term | Meaning |
|---|---|
| **Hardware calibration** | Model-free, serializable backend-performance evidence |
| **Model assessment** | Native evaluation of one exact target at one exact serving profile |
| **Assessing** | Ephemeral observable state while an admitted assessment scope is alive |

## Hardware calibration

Calibration runs bounded synthetic backend operations and records, per backend/device/tensor/workload
class:

- effective bytes per second;
- launch and synchronization cost;
- sample count, duration, dispersion, and stability;
- calibration-method identity.

It reads no model and produces no placement, memory, context, compatibility, or token-rate result.

### Startup contract

```text
native runtime -> hardware topology -> planning-worker pool
               -> cached or measured hardware calibration -> Ready
```

`Ready` guarantees complete calibration for every enabled assessment backend and an operational
planning-worker supervisor. Startup initializes only the worker used for calibration. Additional
workers are created on assessment demand. Progress names the actual CPU, Metal, CUDA, or Vulkan
backend.

### Cache identity

Calibration is atomically cached by:

- method and policy;
- native build and backend ABI;
- enabled backend modules and runtime capabilities;
- normalized hardware topology and required metric coverage.

Corrupt, incomplete, expired, or mismatched evidence is a cache miss. Live free memory and elapsed
calibration wall time are not identity inputs. Model-assessment identity includes the calibration
metric digest.

## Model assessment

`POST /v1/models/assess` accepts batches of exact targets and explicit profiles. A target is one
package or an ordered target/draft pair. A profile currently specifies maximum context for one
sequence.

An uncached assessment performs native work:

1. open GGUF metadata and tensor directories;
2. construct a no-allocation native model;
3. construct a no-allocation context graph for each missing profile;
4. run native placement selection when the requested placement does not fit;
5. combine exact workload facts with hardware calibration.

It reads no tensor payload, allocates no model weights or KV cache, and runs no inference benchmark.
It is still nontrivial: model and context-graph construction are not metadata arithmetic.

```text
cost = one initial model open per target batch
     + one context graph per missing profile
     + native placement-search work where required
```

The initial model open is shared by all profiles for one target. Some native fallback-placement paths
may reopen the model per profile. Different targets cannot share a model object.

### Results

Every requested profile produces one result:

| Result | Meaning |
|---|---|
| `Fits` | Configuration identity, memory accounting, and performance evidence |
| `DoesNotFit` | Configuration identity, memory accounting, limiting resource, and deficit |
| `Incompatible` | The artifact/runtime combination cannot execute |

Malformed or unresolved input produces target-level `InvalidTarget`. Native crash, timeout,
malformed output, impossible arithmetic, or missing output fails the endpoint. Operational failure
is not a model result and creates no cache entry.

## Profiles

ACN requests:

```text
100,000 tokens · 200,000 tokens · exact target maximum
```

Values above the target maximum are removed; the maximum is added; duplicates are removed. A target
below 100,000 is assessed only at its maximum. A pair uses the lower component maximum. ICN does not
search a context range or choose a profile.

## Broad rejection proof

ACN may skip native assessment only when:

```text
exact storage of tensors required by every execution
    > aggregate stable capacity of unique physical memory domains
```

Tensor storage is computed from GGUF tensor shapes and types and deduplicated by immutable content
identity. Optional components are excluded, making this a lower bound on required bytes. Aggregate
stable capacity ignores context, compute, workspace, reserves, and placement constraints, making it
a permissive upper bound. Uncertain targets pass. File size, parameter estimates, model names, and
empirical multipliers cannot reject a target.

## Capacity semantics

Assessment captures one topology and reserve policy. Memory evidence charges model, context,
compute, workspace, projector, target, and draft allocations to canonical physical domains and
device constraints.

Assessment results are validated against the captured topology and capacity policy before reuse.
`Fits`, `DoesNotFit`, and `Incompatible` are completed results and are reusable for the exact
assessment identity. Live availability never participates in assessment cache validity.

Load admission always performs fresh planning against current availability. Cached assessment never
authorizes residency. Explicit stable-capacity native planning may be added only through a proven
binding-level facility; it must not require changes to the nested llama.cpp core.

## Planning-worker pool

ICN owns a small persistent pool:

- capacity is eight workers, subject to available hardware parallelism;
- only the calibration worker exists at startup; other native children are created on demand;
- cold backend initialization is serialized; initialized workers execute concurrently;
- each created worker retains process-local CUDA/Metal/Vulkan state;
- all profiles for one target execute as one worker job;
- different targets use the next available worker concurrently;
- pool size is bounded by hardware parallelism and a fixed safety cap;
- every job has one absolute caller-visible deadline covering queue and native work;
- a failed or timed-out worker is killed and replaced;
- child reaping and diagnostic-reader cleanup cannot delay caller completion.

Inference workers remain separate and model-resident. Backend initialization and ordinary warm-up
may populate driver caches; correctness never depends on cross-process CUDA-context or module sharing.

## Assessing lifecycle

```text
Unassessed -> Assessing -> Assessed
                  |
                  +----> Unassessed on failure, interruption, or deadline
```

`Assessing` is an empty external marker, not a durable record or public operation identity. ACN owns
it inside the scoped assessment Effect:

- enter only after operation admission;
- complete only from that scope's successful result;
- clear in finalization on every exit path;
- never persist it;
- serialize overlapping owners so stale completion is structurally impossible.

ICN independently bounds the complete endpoint and every worker job. A caller finishes at its
deadline even if child termination or reaping stalls.

## Assessment cache and single-flight

The cache unit is one exact profile result. Identity covers:

- immutable target content, roles, and relationships;
- exact serving profile and capacity policy;
- native build, backend ABI, enabled backends, topology, and planning method;
- hardware-calibration metric identity;
- projector, speculative, placement, and execution policy.

ICN checks memory and disk before planner preparation. Missing profiles for one target are batched.
Equivalent target/environment misses share one gate and recheck the cache after admission. Cache
corruption is a miss. Process-local parsed model state is reused only within its batch and is not
serialized.

Stable-topology-checked `Fits`, `DoesNotFit`, and artifact/runtime `Incompatible` results are
persisted. Operational failures are never persisted.

## Product behavior

- Reading catalog, inventory, or TUI state does not itself invoke native assessment.
- One catalog projection assesses recommendable and discovered installed targets through the shared
  assessment service; recommendation policy consumes only recommendable candidates.
- Inventory reconciliation is coalesced background work; reads return the last complete snapshot.
- Installed targets remain visible while assessment is pending or fails.
- Only completed `Fits` configurations become catalog candidates or are eligible for initial
  provider-offering creation.
- Downloading never performs hardware calibration.

## Conformance

- ICN cannot become ready without hardware calibration and an operational worker pool.
- One same-target job returns one result per requested profile.
- Warm exact-cache reads invoke no native planner.
- Warm `DoesNotFit` cache reads invoke no native planner.
- No domain result represents an operational defect.
- `Assessing` cannot survive its owning Effect scope.
- Queueing, native work, caller completion, and child cleanup are all bounded.
- Nested llama.cpp core files remain unmodified.
