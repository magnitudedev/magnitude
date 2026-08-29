---
applies_to:
  - inference/crates/icn-contracts/**
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
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

## Inference acquisition

Chat Completions, Responses, and explicit Instance admission use one residency coordinator.
Inference requests validate the canonical model ID, join or admit residency, wait for readiness,
and atomically acquire a request lease before invoking the backend. There is no ACN preparation,
slot load request, or caller-supplied instance identity.

When `Magnitude-Include-Progress: true` is present, streaming endpoints begin their SSE response
before acquisition and publish meaningful model-loading progress on the same response stream.
Ordinary consumers receive only the standard inference stream.

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
- Client connection or presence state cannot change model residency.
