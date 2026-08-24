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

ICN inserts every admitted Instance into its revisioned registry before acknowledging admission.
Terminal Instances remain as resource-free tombstones for the ICN process lifetime. An Instance
never changes canonical model ID or serving configuration.

Admission for an already Loading or Ready equivalent target joins that occurrence. Conflicting
admission is serialized with replacement. Replacement closes admission to the resident Instance,
waits for accepted inference demand and request leases, stops it, and then loads the successor.
Late transitions cannot reopen a stopping or terminal occurrence.

Caller cancellation after admission detaches that waiter. It does not cancel shared loading or
stop the admitted Instance. Explicit exact-instance stop, idle policy, memory pressure, worker
failure, replacement, or ICN teardown control physical lifetime.

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

Accepted inference demand prevents a conflicting replacement from slipping between readiness and
lease acquisition. Once acquired, the request lease prevents replacement or stop from releasing
the runtime until that request ends.

## First-party projection

The ACN Slot Query returns selection intent only. Client-common composes it with the native Models
and Instances Queries to present availability, residency, and permitted actions. Native resource
invalidations cause those Queries to reread, so agent, CLI, Python, and third-party harness activity
becomes visible without writing through ACN Slot state.

Changing Slot selection never implicitly stops a shared Instance. Explicit warm-load and stop
actions call native Instance operations through the generated inference client and the authored
Effect Query layer.

## Conformance

- ICN is the sole authority for loading, replacement, request leases, idle release, and pressure
  release.
- Every admitted Instance has one canonical model ID and reaches one terminal outcome.
- Equivalent demand joins; conflicting residency is serialized.
- Request cancellation cannot abandon admitted shared work.
- Stop and replacement cannot race past accepted inference demand or active leases.
- ACN never stores or projects Instance residency inside Slot state.
- Agent and external inference requests follow the same ICN acquisition path.
- Loading-instance Stop terminalizes every joined preparation waiter without admitting a
  replacement.
- Ready-instance explicit Stop interrupts active semantic output as `ModelInstanceStopped`.
- Graceful replacement and idle release drain active inference leases.
