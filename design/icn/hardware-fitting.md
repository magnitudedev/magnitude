---
applies_to:
  - inference/crates/icn-hardware/**
  - inference/crates/icn-models/**
  - inference/crates/icn-server/**
  - inference/crates/icn-api/**
  - packages/icn/src/hardware/**
  - packages/acn/src/local-model-evaluations.ts
  - packages/acn/src/local-model-recommendations.ts
  - packages/acn/src/local-provider-offering-projection.ts
  - packages/acn-protocol/src/schemas/model-state.ts
---

# ICN hardware fitting

ICN is the only authority for hardware discovery, native execution planning, memory accounting,
fit assessment, and automatic profile selection. ACN supplies product policy and consumes typed
results. Bun never estimates native memory or reconstructs a native plan.

## Hardware topology

ICN reports physical memory domains, their stable capacity, member devices, and applicable device
limits. Fit uses stable capacity rather than volatile free memory.

The byte vocabulary is explicit: `total_capacity_bytes` is physical capacity,
`usable_capacity_bytes` is the stable fitting capacity after policy reserves, and
`current_available_bytes`/`current_free_bytes` are live observations used only for admission and
serving-time supervision. Cached fitting assessments never use a live observation as capacity evidence.

Unified-memory machines expose one physical memory domain. CPU and accelerator allocations are
charged to that domain once. Device-specific working-set limits remain additional constraints; they
are not reported as independent physical capacity.

Physical-domain identity is canonical across discovery, native fitting, persisted assessment,
model-instance allocation, and clients. The system domain has one reserved identity. Dedicated-domain
identity is derived from exact physical-device evidence and backend-scoped native identity when no
physical identity exists. Device names are presentation only and never domain identity.
Device-specific working-set constraints use a separate canonical native-device identity derived
from normalized backend, physical identity when available, and native ordinal.

One validated memory-topology value owns the binding from every registered native device view to
its physical memory domain. CPU, integrated-GPU, and host-accelerator views are system-memory
bindings; Apple Silicon accelerator views are also system-memory bindings; only genuinely
dedicated devices create dedicated domains. Target, speculative, projector, performance, and
resident-runtime evidence describe only host or native-device allocation locations. A native
observation may omit backend or physical identity while retaining its process-global native index;
blank native strings are missing evidence, not identities. Allocation evidence never classifies a
memory domain. The shared topology resolver performs that classification for all locations, and
inconsistent non-empty evidence or an unregistered index fails closed.

Hardware presentation keeps system-product identity, accelerator chip identity, native backend,
and native device ordinal distinct. Product identity comes from operating-system firmware data;
chip identity comes from the native backend's device description. Generic backend ordinals such as
`CUDA0` and `MTL0` are never interpreted as a particular product or chip.

Hardware observation contains topology, capacity, device limits, and current availability only.
Model identity, model-instance lifecycle and failure, context allocation, parallelism, and
resident-model memory belong to the correlated model-instance residency observation defined by
[local model instance lifecycle](./model-instance-lifecycle.md). Allocation uses canonical
hardware memory-domain identity so presentation can join it to capacity without moving ownership
into the hardware API.

Failure to enumerate or normalize hardware fails the operation. It is never converted into an
empty topology or “no accelerator” result.

## Assessment

`POST /v1/models/assess` accepts a batch of exact model targets and explicit serving profiles.
A target is either one model package or an explicit target/draft pair. A profile supplies maximum
context capacity for one request.

The request also supplies a product memory reserve per physical domain and whether performance
evidence is requested. ICN applies the system-only safety floor defined by
[system memory management](../inference/system-memory-management.md); caller policy may increase
that floor. Dedicated device domains retain the product reserve rather than inheriting the larger
system reserve.

ICN applies that reserve policy when it captures the assessment hardware environment. The
resulting snapshot owns the stable capacities and fingerprint consumed by planning, accounting,
cache validation, and workers. Accounting cannot accept a second reserve policy or recompute
capacity from native fit reports.

Each profile produces one complete result:

- `Fits`, with the ICN-issued serving-configuration identity, memory accounting, and optional
  performance evidence;
- `DoesNotFit`, with the same configuration identity, limiting resource, deficit, and accounting;
  or
- `Incompatible`, with a specific artifact or native-engine diagnostic.

Invalid or incomplete targets are `InvalidTarget` per-target results. Once a target identity is
valid, a planner deadline, process failure, malformed response, output-bound violation, or
performance-calibration failure is an `AssessmentFailed` per-target result with a stable safe code,
message, and retryability. Such a failure does not erase completed sibling results or become model
incompatibility. Failure to capture the shared request environment remains a request-wide failure.
Assessment never installs, configures, offers, selects, or loads a model.

An isolated planner has a five-minute watchdog. This is an operational safety bound, not an
expected latency target. ICN terminates and reaps a worker that reaches the deadline. Worker stdout
and stderr are drained while it runs, retained only to a fixed bound, and never exposed through the
assessment result. Exceeding either output bound terminates the worker as a distinct failure.

Memory accounting is evidence, not a categorical label. Every domain reports its capacity,
required allocation, compatibility reserve, warning reserve, and remaining compatible headroom.
Consumers derive warning presentation from that same accounting; ICN does not publish a parallel
fit label.

## Automatic fitting

`POST /v1/models/fit` selects a profile for each target under explicit bounds:

1. assess exactly one sequence;
2. maximize its complete context length up to the lower of the caller cap, model limit, and
   200,000 tokens; and
3. preserve the effective memory reserve throughout the search.

The result contains the exact `ModelServingConfiguration` and its fitting assessment. A target
that cannot satisfy the minimum context returns `DoesNotFit`; it does not receive an arbitrary
small fallback profile.

ACN uses this operation to create a default offering for an otherwise unconfigured installed
package. Opening a client screen does not start fitting. ACN reconciles installed packages in the
background and retries when relevant hardware topology changes or a prior operational failure may
have cleared.

## Memory meaning

Required memory includes every allocation needed by the exact planned target and profile:

- model weights and mapped or copied buffers;
- context and KV storage;
- compute buffers;
- projector or other auxiliary components; and
- target and draft allocations for speculative decoding.

Each native source is normalized into one charge per owner and allocation location. The charge
carries the complete model, context, compute, and auxiliary memory breakdown. One accountant
resolves each location through the captured topology, aggregates each complete breakdown per
existing domain and device-local limit, and compares only with the stable capacities in that
topology. Fit reports are allocation evidence, never topology evidence.

Product assessment proves the minimum serving guarantee: one sequence with one complete configured
context. It never persists or promises a parallel count.

Performance evidence is advisory recommendation input. It never changes memory fit or authorizes
loading.

## Caching and invalidation

ICN caches assessment results in the shared disposable model-derived cache. A cache key includes
every behavior-changing input:

- immutable package and target identity;
- serving profile;
- reserve and performance policy;
- native-engine, planner, capacity, projector, and speculative-selection fingerprints;
- native build and backend; and
- normalized hardware topology.

A missing, corrupt, or stale entry is a cache miss. On read, the assessment's domain identities and
byte-accounting invariants are validated against the same normalized topology captured for its
environment. An unknown or duplicate domain, a missing system domain, or inconsistent totals
invalidates only that assessment entry. Domain availability must equal the topology's stable
capacity, and each device constraint must identify one current device and exactly match its stable
limit. Cache failure never becomes a model-fit result. ACN may retain product projections, but it
does not persist or recreate ICN assessment evidence.

ICN publishes separate observation streams for all hardware changes and for changes to stable
fitting evidence. Live availability and capture-time changes do not trigger catalog fitting,
automatic setup, or offering reprojection. Those consumers react only to fitting-evidence changes;
admission and hardware presentation continue to consume the full live observation.

Batch assessment captures one normalized hardware environment identity and reuses it for every
target and profile in that request. ICN derives the stable offering-target identity before preparing
model inputs and checks whether every requested profile has valid cached evidence. Complete hits
return without planner-input materialization or native planning. On a miss, an installed target is
planned from its local files and a release-catalog target is planned from installed,
integrity-checked GGUF metadata. A non-catalog remote source target is not an assessable runtime input. Partial hits
prepare the target and compute only the missing profiles.

## Loading

Loading accepts one exact `ModelServingConfiguration`. ICN resolves and reassesses that
configuration under current native-engine and hardware state before allocating, then applies the fresh
availability rule in [system memory management](../inference/system-memory-management.md). A
cached assessment is advisory and cannot authorize a different target, profile, reserve policy,
native-engine build, or topology.

The persistent service sends the exact policy-selected hardware snapshot used for planning and
admission to each isolated planner and inference worker. Workers construct the validated topology
from that supplied snapshot; they do not rediscover or infer memory sharing. A native report that
cannot be resolved against the supplied topology is a topology-change failure, not an invitation
to guess a domain.

After the exact one-sequence baseline fits, loading evaluates native sequence capacities from one
through four. Candidate `P` provisions physical context `configured context × P`, preserves the
baseline target, acceleration, and per-sequence model context, and is selected only when its full
native allocation fits stable capacity and fresh admission. Selected sequence capacity is resolved
execution evidence, not serving configuration identity.

The exact candidate assessments use the shared disposable model-derived cache keyed by model
content, execution profile, capacity policy, native planner/build, backend, and normalized
topology. Preview and load share that evidence. A cache miss performs one batched native planning
pass for the missing candidates; each preview and load still takes a fresh availability sample and
selects the currently admissible candidate.

Once a candidate cache miss enters native planning, the assessor owns that work through cache
publication. Cancellation of an HTTP requester detaches that waiter but does not abandon the
planner subprocess or discard its reusable result. Concurrent misses join through the same cache
coordination boundary when their exact evidence matches; unrelated misses remain concurrent within
the bounded native planner pool.

Successful load evidence identifies the same configuration that was requested. ACN passes the
ICN-issued configuration identity unchanged through recommendation, offering, provider resolution,
slot admission, and model load.

## Product behavior

Installed-package listing reports package and inspection facts without choosing a serving profile.
Fit is meaningful only for a target/profile combination.

The recommendable model catalog supplies targets and eligible profiles. ACN batches assessments,
applies recommendation policy, and publishes recommendations. A persisted provider offering is
projected as available only when all target packages are installed and its exact configuration
currently fits.

## Acceptance criteria

- A target/profile/reserve combination has one ICN-issued configuration identity everywhere.
- ACN contains no native memory estimator or configuration-ID hashing.
- Single-package assessment is never reused for an explicit speculative pair.
- Automatic fitting maximizes exact one-sequence context.
- Loading may select one through four native sequences without changing serving configuration
  identity or per-request context.
- Unified physical memory is never double-counted.
- Target, speculative, projector, performance, and resident allocations use one topology resolver.
- Fit reports and component roles never create or select memory domains.
- Planner and inference workers account against the exact supplied hardware snapshot.
- Discovery, fitting, cached assessment, and residency use the same physical-domain identities.
- Loading reassesses the exact configuration it realizes.
- Deleting assessment caches changes only latency and recomputation.
- Assessing release-catalog targets never requires network access or a remote-header cache.
- One target-specific operational failure cannot fail or discard other results in the same batch.
- Planner diagnostics never cross the assessment API; callers receive stable typed failures.
