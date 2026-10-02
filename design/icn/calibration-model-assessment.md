---
applies_to:
  - inference/service/server/src/assessment/**
  - inference/service/models/src/cache.rs
  - inference/service/models/src/catalog.rs
  - inference/engine/src/assessment.rs
  - inference/engine/executor/src/assessment/**
  - packages/icn-protocol/**
  - packages/acn/src/local-model*.ts
  - packages/acn-protocol/src/schemas/model-state.ts
---

# Assessment environment and model assessment

## Ownership

| Concern | Owner |
| ------- | ----- |
| Device bandwidth, decode costs, per-model memory fit, executability, performance, capabilities | Engine |
| Device selection for assessment (the rule loads use) | Engine |
| Assessment environment identity, targets, pool, cache, deadlines, publication | Service |
| Serving-configuration construction, canonical identity, validation | Service |
| Ranking scores | ACN |
| Presentation | Clients |

## Terms

| Term | Meaning |
| ---- | ------- |
| **Assessment environment** | The selected device, its resolved memory bandwidth and the engine configuration every model is assessed with |
| **Model assessment** | Analytical evaluation of one exact resolved model at its serving profile |
| **Assessing** | Ephemeral observable state while an admitted assessment scope is alive |
| **Dropped** | Terminal disposition for a target whose one assessment attempt failed |

## Assessment environment

The environment is established once, during service start, from the device topology the service
discovers there: the device a native load would select, the host's stable memory, and the selected
device's memory bandwidth (see [performance estimation](performance-estimation.md)). Establishing
it opens no device and is arithmetic over discovered facts; a failure to establish it fails service
start, which the service's own health reports. After start, the environment cannot fail: the pool
has no environment phase, no environment failure and no environment retry.

```text
service start -> device discovery -> environment (selection, host memory, bandwidth)
    -> pool: per-target preparation from headers -> complete assessment
```

## Assessment environment identity

Every cached assessment is keyed by the environment identity, a digest of:

- the engine build (engine version and source digest, covering model families, the fit workload,
  decode costs and the bandwidth table);
- the backend, device selector and device name;
- the resolved bandwidth and its source;
- the normalized stable topology: devices, memory relationships and capacities, without live free
  memory or process-local revisions;
- the process memory limits that bound stable fit capacity;
- the reserve policy (`MemoryReserves`) and the engine serving configuration.

Any change is new work. Old results are unreachable by identity; there is no migration.

## Model assessment

The service derives targets from its current catalog and discovery authorities. A catalog target
selects `Desired` or `Effective` material; a discovery target uses the current ready material.
Material is exact: the release catalog's header bundle before download, installed files after.

One assessment is header arithmetic on the service's bounded blocking pool:

1. the engine opens only the target, projector and separate draft GGUF headers and recognizes the
   family;
2. it derives the model definition, chat capabilities and template fingerprint from its own
   tokenizer, template and reasoning inspection;
3. it resolves the serving configuration (the bundle's declared method, codec, limits) exactly as
   a load does, plans the allocation-free execution plan on the selected device, and builds every
   graph the load prepares, without a device, for the certified memory charge; and
4. it computes memory fit and, for a fitting model, decode speed at every requested depth.

It reads no tensor payload, opens no device, allocates nothing and decodes nothing. A target split
across several GGUF files is assessed as one package: the engine is given its first shard and
reads every shard's header. A speculative bundle's separate draft is interpreted against its
target from its header exactly as a load binds it: its weights, history and draft workflows join
the memory charge, and its decode speed is the target's plain decode (no acceptance is modeled). A
draft the draft family cannot interpret against its target is an unsupported representation; a
declared method the draft does not implement is an invalid configuration, which fails the
assessment as it fails a load. Such a bundle is never assessed or served as plain decoding.

### Executability

The engine can execute a model on a backend exactly when the load would: its family and
representation are recognized, the planner accepts its program, and every kernel call of the
program lies in the kernel's domain (some configuration of the kernel admits the call's static
dimensions). The assessment and the load decide this with the same derivations and classify every
refusal identically:

- the assessment's memory charge builds every graph the load prepares, and a graph node outside
  its kernel's domain fails construction;
- the load checks every kernel it prepares against the kernel's domain before tuning it.

Both report `Unsupported` with the same kernel-domain reason.

### Results

Every assessed profile produces one complete result:

| Result | Meaning |
| ------ | ------- |
| `Fits` | Per-domain memory accounting and one decode-speed estimate per requested depth |
| `DoesNotFit` | Per-domain memory accounting, the limiting domain and its deficit |
| `Unsupported` | The engine cannot execute a discovered model: unrecognized family, unsupported representation (including tokenizer, template and tensor encoding), a planner refusal on the backend, or a kernel call outside its kernel's domain |

A catalog model is never `Unsupported`: an engine `Unsupported` for a catalog target is a release
defect, settled as `Dropped` with an OpenTelemetry error. A family the engine does not implement
carries no capabilities and an empty template fingerprint. Memory domains are named `system` for
host RAM and by device selector for dedicated device memory. Results publish atomically through the
revisioned assessment snapshot; one target's failure never invalidates siblings.

Every exact target receives one attempt. An operational, malformed-material, timeout, or resolution
failure creates no cache entry and settles the target as `Dropped`; it is not re-admitted by a
timer. A dropped discovery target is silent. A dropped reviewed-catalog target emits an
OpenTelemetry error before ACN omits it. A changed artifact, profile, bundle, or environment is
new work, not a retry.

## Profiles

The serving profile's context is the model's supported maximum context as the engine resolves it;
catalog entries declare none. Memory fit uses one conversation at `min(context, 100_000)`.
Performance is sampled at 25K, 50K and 75K where below the context, then at the full context; the
ordered list is nonempty and ends at the context. A result whose engine context differs from its
profile is an operational failure.

## Capacity semantics

Fit compares the standard workload's charge with every domain the load touches: stable
capacity bounded by process limits (and the Metal working set) less that domain's planning
reserve. The charge holds weights and graph memory, plus the larger of two phases that never
coexist: the workload's state at the fit depth, and the load's startup state with its
qualification and import peak. Live availability never participates in assessment identity.
Load admission always plans freshly against current memory; a cached `Fits` never authorizes
residency.

## Assessing lifecycle

```text
assessment admitted -> Assessing -> Fits | DoesNotFit | Unsupported | Dropped
```

`Assessing` is an internal marker owned by the process-lifetime pool:

- enter only while exact work is referenced and admitted;
- complete only from that exact work's result;
- publish an assessed result or dropped disposition on every exit path;
- never persist it;
- guard publication by exact work identity so overlapping reconciliation cannot publish stale
  completion.

## Assessment cache and single-flight

The cache unit is one exact profile result: capabilities, template fingerprint and the profile
result. Its key is the whole assessment identity: environment, exact bundle and profile with its
depths. Equivalent concurrent misses for one bundle and environment share one gate and recheck the
cache after admission. Corruption is a miss. `Fits`, `DoesNotFit` and `Unsupported` are persisted;
operational failures never are. The environment identity includes the engine build, so a fixed
engine is new work.

## Automatic assessment pool

The service maintains one pool over the current catalog and discovered-model sources in its one
environment. Each source slice is `Pending` until its source is read (discovery until its inventory
snapshot is authoritative), then `Available` with one entry per target. Catalog desired material is
admitted immediately when not installed; effective material is used when installed. Ready
discoveries join the same pool without restarting catalog work.

Reconciliation retains terminal evidence, joins equivalent in-flight work, queues missing work, and
cancels work no longer referenced by either source slice. Catalog and discovery expose independent
source revisions. A failed source read is the model inventory's failure, which the catalog reports
from the same listing; the pool keeps that slice as it is and reads the source again after a short
delay. Individual target failures are terminal. Concurrency is bounded by hardware parallelism and
a fixed cap, and every target has one absolute deadline.

## Product behavior

- Reading catalog, inventory, or TUI state does not itself invoke assessment.
- Resolved configurations remain visible while assessment is pending; dropped targets are omitted.
- Only completed `Fits` configurations can become enabled provider offerings; assessment creates no
  durable configuration or installation authority.
- Downloading never measures.
- An incomplete assessment is always genuine progress: models still being assessed, or a source not
  yet read. No assessment failure is shown as progress.

## Conformance

- Assessment reads no tensor payload, opens no device, loads no model and runs no measurement.
- The environment is established before the pool starts; the pool has no environment failure.
- Every `Fits` result contains ordered performance samples at exactly the requested depths.
- No result can state that a model the engine executes cannot run on this computer: `Unsupported`
  comes only from the derivations a load makes.
- Warm exact-cache reads invoke no engine assessment.
- A stale assessment completion cannot overwrite state for a newer exact work identity.
- `Fits`, `DoesNotFit`, and `Unsupported` never represent an operational defect.
- `Assessing` cannot exist without pool-owned queued or running work.
- A settled target cannot return to `Assessing` unless its exact work identity changes.
- ACN contains no assessment scheduler, request correlation, or assessment mutation endpoint.
