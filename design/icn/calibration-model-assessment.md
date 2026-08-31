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
  - web/src/components/local-model-onboarding.tsx
  - web/src/components/model-center.tsx
---

# Hardware calibration and model assessment

## Ownership

| Concern                                                                           | Owner   |
| --------------------------------------------------------------------------------- | ------- |
| Hardware discovery, calibration, native planning, memory and performance evidence | ICN     |
| Serving-configuration construction, canonical identity, validation                | ICN     |
| Profile policy, assessment demand reconciliation, ranking scores                  | ACN     |
| Target filtering, cache reuse, scheduling, concurrency, native work               | ICN     |
| Presentation                                                                      | Clients |

## Terms

| Term                     | Meaning                                                                     |
| ------------------------ | --------------------------------------------------------------------------- |
| **Hardware calibration** | Model-free, serializable backend-performance evidence                       |
| **Model assessment**     | Native evaluation of one exact resolved model at one exact serving profile  |
| **Assessing**            | Ephemeral observable state while an admitted assessment scope is alive      |

## Hardware calibration

Calibration runs bounded synthetic backend operations and records, per backend/device/tensor/workload
class:

- effective bytes per second;
- launch and synchronization cost;
- sample count, duration, dispersion, and stability;
- calibration-method identity.

Calibration attempts every curated weight tensor family supported by the pinned runtime, including
NVFP4, for both dense and routed operations. Unsupported backend/type/operation combinations are
omitted rather than represented by fabricated measurements.

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

`POST /api/v1/catalog/assessments` and `POST /api/v1/discovery/assessments` accept targets addressed
by the current revision of that domain and canonical `ModelId`. A catalog target also selects
`Desired` or `Effective` material. ICN validates the revision and resolves each target to its exact
private servable material before admitting work; packages and bundles do not cross the boundary.
Each requested profile specifies maximum context for one sequence and the context depths at which
performance must be estimated.

An uncached assessment performs native work:

1. open GGUF metadata and tensor directories;
2. construct a no-allocation native model;
3. construct a no-allocation context graph for each missing profile;
4. run native placement selection when the requested placement does not fit;
5. combine the profile's exact workload facts with hardware calibration at every requested
   performance depth.

It reads no tensor payload, allocates no model weights or KV cache, and runs no inference benchmark.
It is still nontrivial: model and context-graph construction are not metadata arithmetic.

```text
cost = one initial model open per target batch
     + one context graph per missing profile
     + native placement-search work where required
```

The initial model open is shared by all profiles for one target. Performance depths within one
profile reuse its single context graph and differ only in estimation arithmetic. Some native
fallback-placement paths may reopen the model per profile. Different targets cannot share a model
object.

### Results

Every requested profile produces one result:

| Result         | Meaning                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| `Fits`         | Exact configuration value, memory accounting, and ordered performance samples |
| `DoesNotFit`   | Exact configuration value, memory accounting, limiting resource, and deficit  |
| `Incompatible` | The artifact/runtime combination cannot execute                               |

The response is a finite generated NDJSON event stream:

```text
Started(environmentId, totalTargets)
Result(result) × exactly totalTargets
Completed(environmentId, totalTargets)
EOF
```

Results are emitted as individual targets finish and may arrive out of request order. Request IDs
provide correlation. `Completed` is the only successful terminal marker; EOF before `Completed` is
an incomplete operation. The stream has no reconnect behavior.

Malformed or unresolved input produces target-level `InvalidTarget`. An operational failure while
assessing one requested target produces `Failed` for that request ID; sibling results remain valid.
`Failed` is not a compatibility or capacity result and creates no cache entry. A failure of shared
admission fails the endpoint before streaming. A later shared failure truncates the stream; results
already emitted remain valid and only unresolved request IDs fail in ACN.

## Profiles

ACN submits the exact profile contained in each issued catalog configuration. Catalog generation rejects a reviewed profile above its artifact
maximum; a pair is bounded by the lower component maximum. Profiles below 4,096 tokens are not
submitted. ICN does not search a context range or choose a profile.

For that one profile, ACN requests performance samples at 25K, 50K, 75K, and full configured
context. Sample depths above the configured context are omitted and duplicates are removed. The
ordered sample list is nonempty and always ends at the full configured context.

## Broad rejection proof

ICN may complete a target without expensive native planning only when:

```text
exact storage of tensors required by every execution
    > aggregate stable capacity of unique physical memory domains
```

Tensor storage is computed from ICN-owned GGUF tensor shapes and types and deduplicated by immutable content
identity. Optional components are excluded, making this a lower bound on required bytes. Aggregate
stable capacity ignores context, compute, workspace, reserves, and placement constraints, making it
a permissive upper bound. Uncertain targets continue through ordinary native planning. ACN performs
no capacity pre-filter. File size, parameter estimates, model names, and empirical multipliers
cannot reject a target.

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

One ICN actor owns a small persistent pool:

- capacity is eight workers, subject to available hardware parallelism;
- only the calibration worker exists at startup; other native children are created on demand;
- configured unused capacity is a number, never a claim that a process exists;
- at most one cold backend activation is in flight; initialized workers execute concurrently;
- expansion has one explicit state: ready, activating, or deferred while a warm worker remains;
  losing the last usable warm worker transitions deferred expansion back to ready, so queued work
  remains eligible for replacement activation;
- each created worker retains process-local CUDA/Metal/Vulkan state;
- all profiles for one target execute as one worker job;
- different targets use the next available worker concurrently;
- pool size is bounded by hardware parallelism and a fixed safety cap;
- every job has one absolute caller-visible deadline covering queue and native work;
- a caller that stops waiting detaches from its reply but does not release or reuse its running worker;
- a failed or timed-out worker replies once, retires, and is replaced only after retirement when demand remains;
- child reaping and diagnostic-reader cleanup cannot delay caller completion.

Inference workers remain separate and model-resident. Backend initialization and ordinary warm-up
may populate driver caches; correctness never depends on cross-process CUDA-context or module sharing.

## Assessing lifecycle

```text
assessment admitted -> Assessing -> terminal result
                                      |-- Fits
                                      |-- DoesNotFit
                                      |-- Incompatible
                                      +-- Failed
```

`Assessing` is an internal marker, not a durable record or public operation identity. ACN owns it
inside the scoped assessment Effect:

- enter only after operation admission;
- complete only from that scope's successful result;
- publish a typed terminal result on every exit path;
- never persist it;
- guard publication by inventory revision and local hardware-cycle generation so overlapping work
  cannot publish stale completion.

ICN independently bounds the complete stream and every worker job. Target work ends before the
stream deadline, reserving time to emit terminal events. A caller finishes at its deadline even if
child termination or reaping stalls.

## Assessment cache and single-flight

The cache unit is one exact bundle/profile result. Identity covers:

- immutable package content, ordered bundle structure, roles, and relationships;
- exact serving profile, requested performance depths, and capacity policy;
- native build, backend ABI, enabled backends, topology, and planning method;
- hardware-calibration metric identity;
- projector, speculative, placement, and execution policy.

ICN checks memory and disk before planner preparation. Missing profiles for one bundle are batched.
Equivalent bundle/environment misses share one gate and recheck the cache after admission. Cache
corruption is a miss. Process-local parsed model state is reused only within its batch and is not
serialized.

Stable-topology-checked `Fits`, `DoesNotFit`, and artifact/runtime `Incompatible` results are
persisted. Operational failures are never persisted.

## ACN demand boundary

`LocalModelAssessor` is ACN's sole native-assessment demand owner. It requests the desired catalog
material when a catalog model is not installed, effective material when installed, and only ready
discovered models. It supplies each model's issued profile; ICN resolves and canonically identifies
the exact private configuration it assesses. Provider, ranking, and product projections consume
the resulting state and do not invoke assessment directly.

One source cycle submits at most one request per nonempty domain. ACN does not batch,
throttle, or capacity-filter the target set; ICN owns target scheduling and native concurrency.
ACN validates `Started`, revision, exact request-ID cardinality, echoed subject and selection,
profiles, environment stability, uniqueness, and `Completed`, and publishes each result immediately
while both source revisions and the hardware-cycle generation remain current.

Source invalidations only request rebuilding and reassessment. Download progress, attempt state,
semantically equivalent inventory observations, catalog presentation, live memory, and client
activity cannot admit assessment. A hardware assessment change admits a new cycle without changing
either source revision.

## Product behavior

- Reading catalog, inventory, or TUI state does not itself invoke native assessment.
- One assessor evaluates issued catalog configurations plus canonical standard profiles through
  the shared assessment service; ICN constructs every resulting exact
  configuration, and ranking policy consumes only private eligible catalog inputs.
- Inventory reconciliation is coalesced background work; reads return the last complete snapshot.
- Resolved configurations remain visible while assessment is pending or fails.
- Only completed `Fits` configurations can become enabled provider offerings; assessment itself
  creates no durable configuration or installation authority.
- Downloading never performs hardware calibration.

## Conformance

- ICN cannot become ready without hardware calibration and an operational worker pool.
- One same-bundle job returns one result per requested profile.
- One reconciliation uses at most one HTTP assessment request per nonempty model domain regardless
  of target count.
- Progress begins at zero, counts exact submitted targets, and advances once per streamed result.
- Stream truncation preserves emitted results and fails only unresolved targets.
- Every profile result carries the exact ICN-constructed serving configuration it assessed.
- Serving configurations cross the boundary as exact values without a separate configuration ID.
- Every `Fits` result contains ordered performance samples ending at the profile context.
- Multiple performance depths for one profile require only one native context graph.
- Warm exact-cache reads invoke no native planner.
- Warm `DoesNotFit` cache reads invoke no native planner.
- Download progress and semantically equivalent source revisions invoke no native assessment.
- A stale assessment completion cannot overwrite state for a newer semantic key.
- No domain result represents an operational defect.
- `Assessing` cannot survive its owning Effect scope.
- ACN contains no broad capacity filter or fixed assessment batch size.
- Queueing, native work, caller completion, and child cleanup are all bounded.
- Nested llama.cpp core files remain unmodified.
