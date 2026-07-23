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
  - packages/protocol/src/schemas/model-state.ts
---

# ICN hardware fitting

ICN is the only authority for hardware discovery, native execution planning, memory accounting,
fit assessment, and automatic profile selection. ACN supplies product policy and consumes typed
results. Bun never estimates native memory or reconstructs a native plan.

## Hardware topology

ICN reports physical memory domains, their stable capacity, member devices, and applicable device
limits. Fit uses stable capacity rather than volatile free memory.

Unified-memory machines expose one physical memory domain. CPU and accelerator allocations are
charged to that domain once. Device-specific working-set limits remain additional constraints; they
are not reported as independent physical capacity.

Hardware presentation keeps system-product identity, accelerator chip identity, runtime backend,
and native device ordinal distinct. Product identity comes from operating-system firmware data;
chip identity comes from the native backend's device description. Generic runtime ordinals such as
`CUDA0` and `MTL0` are never interpreted as a particular product or chip.

Failure to enumerate or normalize hardware fails the operation. It is never converted into an
empty topology or “no accelerator” result.

## Assessment

`POST /v1/models/assess` accepts a batch of exact model targets and explicit serving profiles.
A target is either one model package or an explicit target/draft pair. A profile supplies:

- total shared context capacity; and
- maximum parallel sequence occupancy.

The request also supplies the required memory reserve per physical domain and whether performance
evidence is requested.

Each profile produces one complete result:

- `Fits`, with the ICN-issued serving-configuration identity, memory accounting, and optional
  performance evidence;
- `DoesNotFit`, with the same configuration identity, limiting resource, deficit, and accounting;
  or
- `Incompatible`, with a specific artifact or runtime diagnostic.

Invalid or incomplete targets are per-target results. Operational failures fail the request.
Assessment never installs, configures, offers, selects, or loads a model.

## Automatic fitting

`POST /v1/models/fit` selects a profile for each target under explicit bounds:

1. maximize context length up to the lower of the caller cap, model limit, and 200,000 tokens;
2. after fixing that context, maximize parallel sequences up to the caller cap;
3. preserve the requested memory reserve throughout both searches.

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

Parallel sequences describe occupancy of the shared KV pool. Assessment must follow the pinned
runtime's actual unified or per-sequence KV behavior rather than multiplying a nominal context by
parallelism in Bun.

Performance evidence is advisory recommendation input. It never changes memory fit or authorizes
loading.

## Caching and invalidation

ICN caches assessment results in the shared disposable model-derived cache. A cache key includes
every behavior-changing input:

- immutable package and target identity;
- serving profile;
- reserve and performance policy;
- runtime, planner, capacity, projector, and speculative-selection fingerprints;
- native build and backend; and
- normalized hardware topology.

A missing, corrupt, or stale entry is a cache miss. Cache failure never becomes a model-fit result.
ACN may retain product projections, but it does not persist or recreate ICN assessment evidence.

## Loading

Loading accepts one exact `ModelServingConfiguration`. ICN resolves and reassesses that
configuration under current runtime and hardware state before allocating. A cached assessment is
advisory and cannot authorize a different target, profile, reserve policy, runtime, or topology.

Successful load evidence identifies the same configuration that was requested. ACN passes the
ICN-issued configuration identity unchanged through recommendation, offering, provider resolution,
slot admission, and runtime load.

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
- Context is maximized before parallelism during automatic fitting.
- Unified physical memory is never double-counted.
- Loading reassesses the exact configuration it realizes.
- Deleting assessment caches changes only latency and recomputation.
