---
applies_to:
  - inference/service/server/src/assessment/**
  - inference/service/server/src/worker_process.rs
  - inference/service/models/src/cache.rs
  - inference/service/models/src/catalog.rs
  - inference/engine/src/assessment.rs
  - inference/engine/executor/src/assessment/**
  - packages/icn-protocol/**
  - packages/acn/src/local-model*.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - web/src/components/model-center.tsx
---

# Measurement basis and model assessment

## Ownership

| Concern | Owner |
| ------- | ----- |
| Measurement basis content, per-model memory fit, compatibility, performance, capabilities | Engine |
| Device selection for assessment (the rule loads use) | Engine |
| Measurement job, assessment environment identity, targets, pool, cache, deadlines, publication | Service |
| Serving-configuration construction, canonical identity, validation | Service |
| Ranking scores | ACN |
| Presentation | Clients |

## Terms

| Term | Meaning |
| ---- | ------- |
| **Measurement basis** | The engine's model-free kernel measurements for one execution environment: a fixed plan of operation classes and their cost models |
| **Assessment environment** | The selected device, its basis and the engine configuration every model is assessed with |
| **Model assessment** | Analytical evaluation of one exact resolved model at its serving profile |
| **Assessing** | Ephemeral observable state while an admitted assessment scope is alive |
| **Dropped** | Terminal disposition for a target whose one assessment attempt failed |

## Measurement basis

The engine times a fixed, model-free plan of operation classes on the actual device with shipped
default configurations. There is no search, tuning or per-model measurement: the plan depends only
on the backend, never on which targets exist, so measurement starts at service start and no target
can require a class it lacks.

- **Keys.** A measured key names an operation class and its exact element bindings (the activation
  and, for classes that read weights, the resident weight representation); it carries no model
  geometry. Every representation the backend can keep resident is formed; conversions are keyed by
  their source and destination.
- **Timed and formed entries.** Each class is timed at synthetic sizes chosen to fit its cost model;
  every other binding is only formed, which proves the device executes it. The formed set is the
  compatibility set.
- **Cost models.** A class's points fit one of: a per-launch cost; a line in bytes; a projection cost
  (launch overhead plus seconds per weight byte as a function of output rows, interpolated in log
  rows over a fixed row ladder at a fixed reduction, with a floor point); a history cost (a
  reference-geometry rate and per-axis factors for key-value heads, group and width across depths);
  or a per-byte weight-format factor relative to the reference representation. Only points are
  stored; costs are refitted when the basis is read.
- **Demand.** A model's decode demand maps each term onto a measured key and a shape (plain,
  projection with its weight and launch rows, or attention with its head geometry), so estimation
  is arithmetic over the fitted costs. Demand that maps onto no measured or formed key makes the
  target `Incompatible` naming the key.

A point's variants (arithmetic parameters such as `INT8`, `PARTS`, `SLICES` and `MATRIX`) are screened
with one sample each. On CPU, the cold basis shares one wall-clock target across all still-missing
points, weighted toward history points. Screening probes select the variant; each point receives
at least one subsequent device interval for its fitted cost;
the default and first alternative are screened, while further variants and samples use available
time. The target includes formation and allocation.
On other backends, the default and first alternative are screened and the fastest receives at
least two steady samples. A native submission completes once started, so a single slow operation
may overrun either target. Measurement records every sample actually taken.

- Planned kernels are formed before timing. A dynamically selected extent may require a form
  immediately before its point, never concurrently with its timed submissions.
- CPU timing sizes the independent rotation views to the point's available time. For decode
  attention it uses a 1,024-row reference floor and selects later history depths from the observed
  rate when the requested depth would take too long. Every planned head geometry is still measured,
  and each point records its actual streamed bytes and whether its depth was shortened. One sample
  or a shortened depth supplies limited evidence to downstream confidence.
- Work by other processes on the device is not observable: device times are taken as they fall and
  stored. The basis is an estimate either way; no contention is inferred or corrected.
- A timed submission that faults on the device records the class unsupported, naming the fault;
  models that need it are `Incompatible` with that reason.

### Measurement job

```text
service start -> device discovery -> automatic device selection
    ├── measurement job: form every planned kernel, time the timed entries ──┐
    └── per-target preparation from headers (capabilities, demand, memory) ─┤
                                                               complete assessment
```

- The job and target preparation run concurrently from service start; neither waits for the other.
- Preparation reads only headers: family recognition, the execution plan, chat capabilities
  (tokenizer configuration validated without building a tokenizer, template inspected once per
  template), decode demand and the certified memory charge. It is milliseconds of work per target.
- One contained child process of the service executable opens exactly the selected device, loads
  a stored basis for that device's measurement identity when one is complete, otherwise measures
  the plan, stores the basis and reports its identity. The service process never opens a device,
  forms kernels or times them. The child checkpoints completed classes during a cold measurement;
  after interruption, a new job measures only missing classes. The child's output streams are read
  to their end, including after a deadline kills the child. Failure and deadline results retain
  complete worker diagnostics; release acceptance also records the raw stream independently.
- The assessment pool is `Preparing` until the basis is available. A failed job publishes a
  retryable pool failure and is retried with bounded backoff; it never blocks service health.
- Measurement and model residency exclude each other on the device: measurement waits for no
  instance to be loading or resident, and loads wait for measurement to finish.
- Measurement allocations are engine claims above the planning reserve.
- A changed engine, driver or toolchain, device or plan yields a new measurement identity and
  therefore a new measurement; a basis is never revalidated or migrated.

## Assessment environment identity

Every cached assessment is keyed by the environment identity, a digest of:

- the engine build (engine version and source digest, covering model families, the fit workload
  and the kernel bundle);
- the backend and its toolchain identity (Seismic's device tuning identity);
- the device selector and the normalized stable topology: devices, memory relationships and
  capacities, without live free memory or process-local revisions;
- the process memory limits that bound stable fit capacity;
- the measurement basis digest and protocol version;
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
   a load does and plans the allocation-free execution plan on the selected device; and
4. after the basis is ready, it computes memory fit, compatibility against the basis and decode
   speed at every requested depth.

It reads no tensor payload, opens no device, allocates nothing and decodes nothing. A target split
across several GGUF files is assessed as one package: the engine is given its first shard and
reads every shard's header. A speculative bundle's separate draft is interpreted against its
target from its header exactly as a load binds it: its weights, history and draft workflows join
the memory charge, and its decode speed is the target's plain decode (no acceptance is modeled, so
drafts add no demand terms outside the fixed basis). A draft the draft family cannot interpret
against its target is an unsupported representation; a declared method the draft does not
implement, or a draft the executor cannot run, makes the bundle `Incompatible`. Such a bundle is
never assessed or served as plain decoding.

### Results

Every assessed profile produces one complete result:

| Result | Meaning |
| ------ | ------- |
| `Fits` | Per-domain memory accounting and one performance sample per requested depth |
| `DoesNotFit` | Per-domain memory accounting, the limiting domain and its deficit |
| `Incompatible` | Unsupported family, representation (including tokenizer, template and tensor encoding), backend, or operation outside the basis |

There is no unknown, unconfirmed or partial result. A family the engine does not implement carries
no capabilities and an empty template fingerprint. Memory domains are named `system` for host RAM
and by device selector for dedicated device memory. Results publish atomically through the
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

Fit compares the standard workload's clean-load charge with every domain the load touches: stable
capacity bounded by process limits (and the Metal working set) less that domain's planning
reserve. Live availability never participates in assessment identity. Load admission always plans
freshly against current memory; a cached `Fits` never authorizes residency.

## Assessing lifecycle

```text
assessment admitted -> Assessing -> Fits | DoesNotFit | Incompatible | Dropped
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
cache after admission. Corruption is a miss. `Fits`, `DoesNotFit` and `Incompatible` are
persisted; operational failures never are.

## Automatic assessment pool

The service maintains one pool over the current catalog and discovered-model sources. Catalog
desired material is admitted immediately when not installed; effective material is used when
installed. Discovery remains pending until its inventory snapshot is authoritative, then ready
discoveries join the same pool without restarting catalog work.

Reconciliation retains terminal evidence, joins equivalent in-flight work, queues missing work, and
cancels work no longer referenced by either source slice. Catalog and discovery expose independent
source revisions. Whole-source reconciliation failures are retried with bounded background backoff.
Individual target failures are terminal. Concurrency is bounded by hardware parallelism and a fixed
cap, and every target has one absolute deadline.

## Product behavior

- Reading catalog, inventory, or TUI state does not itself invoke assessment.
- Resolved configurations remain visible while assessment is pending; dropped targets are omitted.
- Only completed `Fits` configurations can become enabled provider offerings; assessment creates no
  durable configuration or installation authority.
- Downloading never measures.

## Conformance

- Assessment readiness waits for the measurement basis; service health never does.
- Measurement never overlaps a loading or resident instance on the device.
- Assessment reads no tensor payload, opens no device, loads no model and runs no per-model
  measurement.
- Every `Fits` result contains ordered performance samples at exactly the requested depths.
- Warm exact-cache reads invoke no engine assessment.
- A stale assessment completion cannot overwrite state for a newer exact work identity.
- `Fits`, `DoesNotFit`, and `Incompatible` never represent an operational defect.
- `Assessing` cannot exist without pool-owned queued or running work.
- A settled target cannot return to `Assessing` unless its exact work identity changes.
- ACN contains no assessment scheduler, request correlation, or assessment mutation endpoint.
