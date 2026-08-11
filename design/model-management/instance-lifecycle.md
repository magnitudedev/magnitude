---
applies_to:
  - inference/crates/icn-contracts/**
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
  - packages/icn/src/instances/**
  - packages/icn/src/provider/**
  - packages/acn/src/model-*.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/utils/**
  - cli/src/features/local-inference/**
---

# Model instance lifecycle

This document defines the boundary between durable ACN model slots and physical ICN model
instances. Terms follow [Model-management terminology](./terminology.md).

## Ownership

```text
ACN ModelSlot                         ICN ModelInstance
durable product intent               physical runtime truth
provider offering and selection      one admitted occurrence
availability and user actions        allocation and native worker lifecycle
optional instance projection         authoritative instance state
```

Hardware observation owns topology, capacity, and current availability. It owns neither slot nor
instance lifecycle. A model instance never becomes durable selection merely because it is resident.

## Identity

A slot is identified by `SlotId`. A serving configuration identifies reusable execution intent. A
model instance identifies exactly one physical occurrence of that configuration. Reusing a
configuration creates a new instance identity; reusing an active instance identity for another
configuration is invalid.

Commands and late events address the narrowest identity required. A stop addresses an exact
instance. A request binds to both instance and configuration so an ended occurrence cannot affect or
receive work intended for a successor.

## Native lifecycle

```text
Loading -> Ready -> Stopping -> Stopped
    |         |         |
    +---------+---------+-> Failed
```

ACN creates the instance identity before requesting load. Repeating the same instance/configuration
pair is idempotent; reusing the identity for different data is a conflict. ICN publishes every
admitted transition and exactly one terminal outcome. Caller interruption after admission detaches
the waiter but does not abandon instance ownership. Only explicit stop, autonomous safety policy,
failure, or ICN teardown terminates an admitted instance.

ICN retains resource-free terminal tombstones for the controller lifetime. The complete revisioned
instance snapshot includes them, so exact-ID observation and idempotent replay cannot lose an
accepted occurrence. Watch is coalescing invalidation; observers admit Watch before Get, retry read
failure, re-admit terminated watches, and never convert failure into an empty snapshot.

Loading validates exact package presence and inspection, resolves a fresh native plan, and applies
current memory admission before allocation. It first proves the one-sequence baseline, then chooses
the greatest admissible sequence-capacity option from one through four. The chosen capacity and
physical context are instance evidence, not serving-profile or configuration identity.

Stopped instances retain one release reason: `user_stop`, `idle_timeout`, `replacement`, or
`memory_pressure`. Controller teardown destroys the authority and is not an instance release
reason. Once stopping begins, late loading or ready publications cannot reopen the instance.

## Slot aggregate

A slot projects:

- durable selection and reasoning effort;
- provider-offering availability;
- desired and actual configuration identity;
- an optional exact instance and native allocation;
- valid actions derived from authoritative state; and
- typed failure owned by the operation or instance that produced it.

Slot state never invents a synthetic instance while loading and never treats provider availability
as proof of residency. Multiple slots selecting the same configuration may share one compatible
ready instance when policy permits; each slot retains independent durable intent.

Slot transitions use the typed slot FSM. One aggregate commit publishes durable selection and the
agent model configuration atomically, while each observer advances only on semantic change. The
aggregate privately retains the exact configuration and admitted instance identity used by
commands, avoiding a second offering lookup after commit.

An unloaded configured slot may be previewed only by an explicit observational load-plan query.
Preview does not mutate, persist, authorize assignment, or authorize loading; actual load repeats
planning and admission.

## Request acquisition

Before dispatch, ACN resolves the selected provider offering and acquires an exact ready instance.
Equivalent concurrent acquisition joins compatible admitted work rather than spawning duplicate
loads. A request lease protects the instance and required packages for the request lifetime.

The lease is an internal ICN capability acquired atomically against instance ID, configuration ID,
Ready state, open admission, and the owned backend. Stop and replacement close admission before
draining accepted leases. Every worker, package claim, allocation, and load operation is attributed
to exactly one instance entry.

The provider request names the exact instance and configuration. ICN rejects stale, mismatched,
stopping, stopped, or failed instances. Request cancellation releases only that request's lease and
does not imply instance stop.

Equivalent concurrent ACN load commands join one finite command keyed by exact configuration while
any selecting slot still needs it. This is command deduplication, not another physical lifecycle or
admission gate. A superseded command stops only the exact instance it admitted.

## Stop, failure, and pressure

Explicit stop is idempotent for its exact instance. ICN prevents new request acquisition, drains or
terminates work according to the stop reason, releases native resources, and publishes terminal
state before releasing live ownership. The terminal tombstone remains observable for the controller
lifetime.

Memory-pressure eviction uses the same physical memory domains and allocation evidence as
assessment and load admission. It may stop an instance but does not clear the durable slot
selection. Unexpected worker exit becomes typed instance failure and cannot be mistaken for a
successful stop or provider incompatibility.

After restart, ACN restores retained configurations and slot selections, projects their current
provider offerings, and does not reconstruct prior instance identity or residency. New demand
creates a new admitted occurrence.

## Conformance

- Slot selection, configuration, and instance identities are never interchangeable.
- Every admitted instance reaches one authoritative terminal outcome.
- Loading repeats current package, topology, planning, and admission validation.
- Late events from an old instance cannot mutate its successor.
- Terminal tombstones preserve exact-ID observation and idempotent replay.
- Request leases prevent removal of resources still in use.
- Eviction and failure preserve durable user selection.
- Restart never claims that a pre-restart instance remains resident.
