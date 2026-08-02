---
applies_to:
  - packages/event-core/src/ambient/**
  - packages/event-core/src/core/ambient-service.ts
  - packages/event-core/src/core/projection-bus.ts
  - packages/event-core/src/projection/**
---

# Ambient transactions

Ambients are process-local runtime facts that may change without creating persisted events.
Event-core owns the consistency boundary between an ambient value and projections derived from it.

An ambient update is one serialized transaction:

1. commit the new ambient value;
2. run every registered ambient handler in projection dependency order;
3. flush signals and runtime observers caused by those handlers; and
4. release projection readers.

Ambient transactions, event projection processing, and other ambient transactions do not
interleave. Projection handlers therefore observe one ambient snapshot while reducing an event or
ambient change.

Updates are admitted to an event-core-owned queue. Once admitted, processing is owned by the
event-engine scope rather than the calling fiber: interrupting a caller stops its wait but does not
abandon a partially applied transaction. Engine shutdown may interrupt the queue because no
projection reads survive that scope.

An ambient update invoked synchronously by a projection handler is processed reentrantly within the
active transaction. It does not wait behind itself or release the serialization boundary between
the nested update and outer event or ambient handler.

Ordinary projection accessors wait for an active ambient transaction to finish before returning
state. Runtime consumers hold the same read boundary through snapshot construction and dependency
registration, so a transaction cannot publish between a consumer read and its subscription update.
Their invalidation subscription is active before the initial snapshot begins, so publication after
that boundary cannot be lost while the consumer transitions from its snapshot to its change stream.
Code already executing inside that transaction may use projection reads without waiting on itself.
Raw subscription references remain observation primitives rather than transactional read boundaries.

Ambient domain values do not carry synchronization revisions. Transaction coordination is
process-local event-core state and is not serialized into events, projections, or protocol
contracts.

This guarantee is implemented behind the existing ambient update and projection accessor
interfaces. It does not add a consumer-facing synchronization API.

## Acceptance criteria

1. An event handler cannot run between an ambient value commit and its projection handlers.
2. A projection accessor invoked during an ambient update does not return the pre-update projection
   after the new ambient value is committed.
3. Signals and runtime observers caused by an ambient handler complete before the transaction is
   released.
4. Ambient updates remain non-events and introduce no persisted or public revision field.
5. Ambient handlers can use their declared synchronous reads without deadlocking the transaction.
6. Interrupting an update caller after admission does not abandon the admitted transaction.
7. A projection handler can update a registered ambient without deadlocking.
