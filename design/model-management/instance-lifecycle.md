---
applies_to:
  - inference/service/api/**
  - inference/service/server/src/residency/**
  - inference/service/server/src/configurations.rs
  - inference/service/server/src/serving.rs
  - inference/service/contracts/**
  - packages/icn/src/instances/**
  - packages/icn/src/provider/**
  - packages/acn/src/model-*.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/sdk/src/inference*
  - packages/client-common/src/model-slots/**
---

# Model instance lifecycle

This document defines the boundary between durable Magnitude model selection and physical ICN
model residency.

## Ownership and identity

ACN Slots contain durable user intent: provider, canonical provider model ID, reasoning preference,
favorites, and recency. They contain no physical residency, instance binding, or load command.

ICN owns Model Instances. An Instance is one process-local admitted occurrence of the canonical
model ID named by callers. It records its ICN-created instance ID, canonical model ID, resolved
serving configuration, lifecycle progress, and ready allocation. Only the instance ID identifies
that physical occurrence; it is never persisted as harness or Slot model selection.

The canonical model ID, such as `gemma-4-26b-a4b-it-qat:gguf:q4`, is the only callable model
identity. ICN does not mint aliases or wrapper identities.

## Load plan and allocation

A read-only load plan, also carried by a Loading Instance and a Stopping Instance that never
became resident, states the serving context window (the engine's supported maximum), the device
the load would use (its device identity and backend), and the required memory: the load's startup
peak claim in that device's allocation domain, system RAM or dedicated VRAM. Required memory
excludes future context growth, which the engine claims as history grows.

A Ready allocation states the serving context window and, per memory domain, the engine heap's
observed model, context, compute, and auxiliary bytes at the last observation. Memory is elastic,
so this is standing, not a fixed reservation. No plan or allocation carries fixed sequence slots
or a physical context size.

A load that fits the machine but cannot be given its memory fails as a memory shortage, in one
of two forms. Blocked: memory that cannot be moved is in the way, and the failure reports the
required memory and what a load may claim in the limiting domain, the reserve already taken out.
The limiting domain is the one whose claim the engine refused: the device's allocation domain, or
system RAM when a dedicated device's staged upload did not fit. Under pressure: a domain the load
uses was in the Reclaim band when the load claimed from it or while it imported weights, and no
byte count describes it. Clients present both as one condition and never show a reserve.

A load requested while admission is closed waits in the queued stage and starts when admission
reopens. An Instance released for memory pressure stays distinguishable from any other unloaded
Instance, so a client can say why the model is no longer loaded.

## Lifecycle

```text
Loading -> Ready -> Stopping -> Stopped
    |         |         |
    +---------+---------+-> Failed
```

One ICN residency actor owns the singleton physical slot, its current Instance, owned load or
release operation, FIFO conflicting demand, Ready worker and leases, revision, and terminal
tombstones. No registry, Ready mirror, runtime record, or transition-wide mutex independently
represents the same residency. The actor publishes an admitted Loading Instance before its child
load can report progress. Terminal Instances remain as resource-free tombstones for the ICN
process lifetime. An Instance never changes canonical model ID.

Admission for an already Loading or Ready equivalent model ID joins that occurrence. Conflicting
admission enters one actor-owned FIFO. Replacement closes admission to the resident Instance,
drains its actor-issued request leases, stops it, and only then admits the successor. Child results
carry the exact instance ID; late results are discarded with their owned resources and cannot
reopen a stopping or terminal occurrence.

Caller cancellation after admission detaches that waiter. It does not cancel shared loading or
stop the admitted Instance. Explicit exact-instance stop, idle policy, memory pressure, worker
failure, replacement, or ICN teardown control physical lifetime.

The idle policy is fixed and client-independent. One actor-owned monotonic deadline exists only
while the Instance is Ready with zero inference leases. Readiness without leases, final lease
release, and equivalent explicit warm demand each start a full one-hour interval. Acquiring any
lease clears the deadline. Expiration is serialized with inference admission and rechecks Ready
state and zero leases before beginning graceful `IdleTimeout` release.

An explicit Stop during Loading transitions the exact occurrence through Stopping to
`Stopped(UserStop)`. Every inference waiter joined to that occurrence receives the same canonical
non-retryable `model_instance_stopped` result. A Ready occurrence has active inference leases;
explicit Stop closes admission, terminates active execution, and reports the canonical
non-retryable `model_instance_stopped` result to every affected stream. Graceful replacement and
idle release, rather than explicit Stop, drain active leases.

Completion of a worker's release operation is not itself retirement evidence. Canceled loads,
failed loads, and failed Ready workers retain their worker through cleanup. The actor verifies the
exact worker's retained exit result before publishing a resource-free terminal state or admitting a
successor. Failure cleanup uses `Stopping(Failure)` and preserves the original failure for its
eventual Failed tombstone, including when an explicit Stop retries cleanup.
An unproven release remains Stopping with its worker owned: Stop and queued demand receive a typed
failure, new loads and package removal are rejected, and a later exact-instance Stop retries that
same worker. Native exit evidence persists after reaping; absence of a live PID is not used as proof.

## Loading on the engine worker

A resident model runs in one `inference-worker` process of the service executable; the service
owns its spawn, its framed transport, crash handling and the proof of its retirement. The host
process keeps the model's chat semantics.

A load resolves the model's installed material through the service's single resolved-configuration
cache, which host-only operations (counting, template application, properties) share, so no second
tokenizer, template or properties path exists and those operations never lease or load. The load
then waits while memory admission is closed (stage `queued`), previews itself on the service's
device catalog (the same engine preview the load-plan endpoint returns), and has the worker load
exactly the previewed device with the service's kernel cache and reserve policy.

`Loading` carries a `stage` and a `fraction`, both measured from the worker's work and never from
time. The stages are `queued`, `preparing` (resolution, planning, opening the device, preparing
programs), `optimizing` (kernel tuning), `loading_weights` and `finalizing` (state allocation,
warm-up, readiness verification). Before tuning begins the worker counts its tuning units and
finds each one's stored result, so a load enters `optimizing` only when it will search, and knows
this before any tuning or weight import starts. Tuning progress is the configuration budget of
the searched units over that of every unit that searches; weight progress is resident bytes
imported over the target's. A load that tunes fills the fraction's first half with tuning and the
second with weights; one that does not fills it with weights. The fraction is monotonic, stays at
zero until measured work starts, and holds through `finalizing`; only `Ready` means the load is
complete. Progress is published in bounded steps rather than per unit or weight. Readiness is verified before the Instance is
Ready: the worker must report the package identity the host resolved, the chat-template
fingerprint and input modalities it read from its own opened package equal to the host's, and the
previewed device. The
Ready allocation is the worker's allocation census, republished when it changes, at most once per
second.

A catalog installation's optimization (see
[catalog and acquisition](./catalog-and-acquisition.md)) moves a new model's tuning ahead of its
first load. It resolves the configuration through the same resolved-configuration cache, previews
it to find the device a load would select (a model whose preview fails cannot load, so its
optimization is skipped), shares the device with loads while excluding assessment measurement
exactly as a load does, and runs an
inference worker with a prepare-only request for exactly that device and the same kernel cache.
Its worker reports the load's `preparing` and tuning progress, stores each tuned unit as it
completes, and exits once prepared; it never creates an Instance, holds residency, or serves.
There is at most one preparation job per servable bundle. A load of that bundle first stops the
job and proves its worker retired, then spawns its own worker, which tunes only what is not yet
stored. Other models may be resident and serving while a job runs. A job that fails (memory
exhaustion beside a resident model, the tuning safety stop, worker loss) is logged and never
reported as a model failure.

Every release first asks the worker to shut down, which ends its open requests as
`model_instance_stopped`. Graceful release (replacement, idle) allows two seconds and explicit Stop
half a second before the worker is killed. Retirement is proven by the worker's exit status.

Memory pressure has two sources with one outcome. The engine unloads itself when a memory domain
it uses stays in the Reclaim band (headroom at or below its planning reserve, or host distress),
and the service kills the worker on the first system-RAM sample at or below the emergency reserve
or at critical kernel pressure. Either releases the Instance as `memory_pressure` and closes load
admission until system RAM has stayed out of distress with headroom above the planning reserve for
five seconds; a failed sample restarts that wait. Nothing reloads automatically. Worker
exit, a lost device, and one continuous second of failed memory observation fail the Instance.

## Inference acquisition

Chat Completions, Responses, and explicit Instance admission use one residency coordinator.
Inference requests validate the canonical model ID, join or admit residency, wait for readiness,
and atomically acquire a request lease before invoking the backend. There is no ACN preparation,
slot load request, or caller-supplied instance identity.

When `Magnitude-Include-Progress: true` is present, streaming endpoints begin their SSE response
before acquisition and publish model-loading progress on the same response stream: the Instance's
`stage` and `fraction` under the same names and values as the Instance status.
Ordinary consumers wait through acquisition and inference admission before opening a successful
stream. They receive only the standard inference stream.

Loading inference waiters belong to the Loading state. Loading success moves the worker to Ready
and grants live waiters actor-issued leases in the same mailbox transition, so no pending-demand
counter is needed. Once acquired, a lease makes replacement and idle release drain. Exact explicit
Stop instead interrupts the worker and every active lease-dependent request immediately.

## First-party projection

The ACN Slot Query returns selection intent. The ACN Catalog projects current ICN Instance truth
into each local model's installed-family acquisition state, so clients receive availability,
residency, and permitted actions without consuming native resources or joining separate model and
instance snapshots. Native invalidations cause ACN to reread and reproject, so agent, CLI, Python,
and third-party harness activity becomes visible without writing through Slot state.

Instance changes also invalidate the live Hardware snapshot and load preview because resident
allocation changes current headroom. They do not invalidate stable model assessment, which is
defined against stable capacity rather than live availability.

Changing Slot selection never implicitly stops a shared Instance. Slot-oriented clients may ask
ACN to load or stop the model selected by a configured Slot. The non-interactive CLI loads by
canonical model ID and exposes one zero-argument stop for the singleton active Instance. ACN
resolves either product intent to ICN's exact Instance operations; callers never supply a native
Instance ID.

## Conformance

- ICN is the sole authority for loading, replacement, request leases, idle release, and pressure
  release.
- Every admitted Instance has one canonical model ID and reaches one terminal outcome.
- Equivalent demand joins; conflicting residency is serialized.
- Request cancellation cannot abandon admitted shared work.
- Stop and replacement cannot race past accepted inference demand or active leases.
- ACN never stores Instance residency inside Slot state; it projects current residency into the
  client-facing local model.
- Agent and external inference requests follow the same ICN acquisition path.
- Loading-instance Stop terminalizes every joined preparation waiter without admitting a
  replacement.
- Ready-instance explicit Stop interrupts active semantic output as `ModelInstanceStopped`.
- Graceful replacement and idle release drain active inference leases.
- A worker loads exactly the device its load previewed; readiness verifies the package identity
  and device against the host's resolution.
- Host-only operations never lease or load a model.
- A post-installation preparation never holds residency; a load of the same bundle stops it, and
  its tuned units remain stored for that load.
- Engine unload for memory pressure and the service's emergency kill both publish
  `memory_pressure` and gate new loads on five seconds without host distress and with headroom
  above the planning reserve.
- Client connection or presence state cannot change model residency.
