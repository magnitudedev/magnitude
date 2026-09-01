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
| Profile policy and assessment demand reconciliation                              | ICN     |
| Ranking scores                                                                    | ACN     |
| Target filtering, cache reuse, scheduling, concurrency, native work               | ICN     |
| Presentation                                                                      | Clients |

## Terms

| Term                     | Meaning                                                                     |
| ------------------------ | --------------------------------------------------------------------------- |
| **Hardware calibration** | Model-free, serializable backend-performance evidence                       |
| **Model assessment**     | Native evaluation of one exact resolved model at one exact serving profile  |
| **Assessing**            | Ephemeral observable state while an admitted assessment scope is alive      |
| **Dropped**              | Terminal disposition for a target whose one assessment attempt failed       |

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

ICN derives targets from its current catalog and discovery authorities. A catalog target selects
`Desired` or `Effective` material; a discovery target uses the current ready material. ICN resolves
each target to exact private servable material before admitting work. Each profile specifies
maximum context for one sequence and the standard context depths at which performance is estimated.

An uncached assessment performs one worker job:

1. open the target Assessment Material once as a no-allocation native model;
2. derive the effective-template fingerprint and tool/reasoning capabilities from that model;
3. derive projector modalities and resolve speculative-decoding compatibility;
4. reuse that same model handle to construct a no-allocation context graph for each missing profile;
5. run native placement selection when the requested placement does not fit;
6. combine the profile's exact workload facts with hardware calibration at every requested
   performance depth.

It reads no tensor payload, allocates no model weights or KV cache, and runs no inference benchmark.
It is still nontrivial: model and context-graph construction are not metadata arithmetic.

```text
cost = one initial model open per target batch
     + one context graph per missing profile
     + native placement-search work where required
```

The model open is shared by template analysis and all profiles for one target. Performance depths
within one profile reuse its single context graph and differ only in estimation arithmetic. Some
native fallback-placement paths may reopen the model per profile. Different targets cannot share a
model object.

### Results

Every assessed profile produces one result:

| Result         | Meaning                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| `Fits`         | Exact configuration value, memory accounting, and ordered performance samples |
| `DoesNotFit`   | Exact configuration value, memory accounting, limiting resource, and deficit  |
| `Incompatible` | The artifact/runtime combination cannot execute                               |

The terminal assessed state is flat: it contains capabilities, the template fingerprint, and the
ordered profile results. Its cache entry also retains the native template/reasoning configuration
and resolved speculative configuration required by residency. Results are published atomically
through the revisioned automatic assessment snapshot. Each available source
slice contains its current exact subjects. A target is `Assessing`, `Assessed`, or `Dropped`; total
and settled counts are derived from that list rather than stored as parallel state. One target
failure does not invalidate sibling results.

Every exact target receives one assessment attempt. Any operational, malformed-material, timeout,
or resolution failure creates no cache entry and settles that target as `Dropped`; it is not
re-admitted by a timer. A dropped discovery target is silent. A dropped reviewed-catalog target
emits an OpenTelemetry error before ACN omits it from the product projection. A changed artifact,
profile, bundle, hardware environment, or other exact work identity is new work, not a retry.
Observation is read-only and has no request, correlation, or stream lifecycle.

## Profiles

ICN assesses the exact profile contained in each current catalog or discovered configuration.
Catalog generation rejects a reviewed profile above its artifact maximum; a pair is bounded by the
lower component maximum. ICN does not search a context range or choose a replacement profile.

For that profile, ICN derives performance samples at 25K, 50K, 75K, and full configured context.
Sample depths above the configured context are omitted and duplicates are removed. The ordered
sample list is nonempty and always ends at the full configured context.

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

Load admission always performs fresh memory planning against current availability, using the
completed assessment's capability and speculative evidence. It does not repeat template,
projector, or speculative capability work. The loaded runtime verifies fingerprint and modalities
against that evidence. Cached assessment never authorizes residency by itself. Explicit
stable-capacity native planning may be added only through a proven binding-level facility; it must
not require changes to the nested llama.cpp core.

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
                                      +-- Dropped
```

`Assessing` is an internal marker, not a durable record or public operation identity. ICN owns it
inside its process-lifetime assessment pool:

- enter only while exact work is referenced and admitted;
- complete only from that exact work's result;
- publish an assessed result or dropped disposition on every exit path;
- never persist it;
- guard publication by exact work identity so overlapping reconciliation cannot publish stale
  completion.

ICN independently bounds every worker job. Observation is snapshot-based and does not own or wait
for assessment work.

## Assessment cache and single-flight

The cache unit is one exact Assessment Material/profile result containing public capabilities,
native template/reasoning configuration, template fingerprint, resolved speculative configuration,
and profile evidence. Identity covers:

- immutable package content, ordered bundle structure, roles, and relationships;
- exact serving profile, requested performance depths, and capacity policy;
- native build, backend ABI, enabled backends, topology, and planning method;
- hardware-calibration metric identity;
- projector, speculative, placement, and execution policy.

ICN checks memory and disk before planner preparation. Missing profiles for one bundle are batched.
Equivalent bundle/environment misses share one gate and recheck the cache after admission. Cache
corruption is a miss. The no-allocation native model handle is reused only within its worker job and
is not serialized.

Stable-topology-checked `Fits`, `DoesNotFit`, and artifact/runtime `Incompatible` results are
persisted. Operational failures are never persisted.

## Automatic assessment pool

ICN maintains one assessment pool over the current catalog and discovered-model sources. Catalog
desired material is admitted immediately when not installed; effective material is used when
installed. Discovery remains pending until its atomic inventory snapshot is authoritative, then
ready discoveries are added to the same pool without restarting catalog work.

Work identity includes the exact resolved bundle, profile, performance depths, and assessment
environment. Reconciliation retains terminal evidence, joins equivalent in-flight work, queues
missing work, and cancels work no longer referenced by either current source slice. Publication is
guarded by that exact identity, so an obsolete completion cannot update a superseding target.
Catalog and discovery expose independent source revisions and progress views over the unified pool.
Whole-source reconciliation failures are retried with bounded background backoff. Individual
target failures are terminal and never retried. Foreground planning takes precedence over queued
background assessment.

## Product behavior

- Reading catalog, inventory, or TUI state does not itself invoke native assessment.
- One ICN-owned pool evaluates current catalog and authoritative discovered configurations; ranking
  policy consumes only private eligible catalog inputs.
- Inventory reconciliation is coalesced, retrying background work; reads remain non-blocking.
- Resolved configurations remain visible while assessment is pending; dropped targets are omitted.
- Only completed `Fits` configurations can become enabled provider offerings; assessment itself
  creates no durable configuration or installation authority.
- Downloading never performs hardware calibration.

## Conformance

- ICN cannot become ready without hardware calibration and an operational worker pool.
- One same-bundle job returns one result per requested profile.
- Starting discovery or assessment never delays ICN or ACN health.
- Discovery adds work to the live pool without restarting unchanged catalog work.
- Progress counts current exact targets and advances once per assessed or dropped target.
- Every profile result matches the current subject's exact ICN-resolved serving configuration;
  private material remains inside ICN.
- Every `Fits` result contains ordered performance samples ending at the profile context.
- Multiple performance depths for one profile require only one native context graph.
- Warm exact-cache reads invoke no native planner.
- Warm `DoesNotFit` cache reads invoke no native planner.
- Download progress and semantically equivalent source revisions invoke no native assessment.
- A stale assessment completion cannot overwrite state for a newer semantic key.
- `Fits`, `DoesNotFit`, and `Incompatible` never represent an operational defect; operational
  defects drop the target.
- `Assessing` cannot exist without pool-owned queued or running work.
- A settled target cannot return to `Assessing` unless its exact work identity changes.
- ACN contains no assessment scheduler, request correlation, or assessment mutation endpoint.
- Queueing, native work, and child cleanup are bounded.
- Nested llama.cpp core files remain unmodified.
