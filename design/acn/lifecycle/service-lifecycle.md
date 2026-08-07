---
applies_to:
  - packages/acn/src/service-lifecycle.ts
  - packages/acn/src/server.ts
  - packages/acn/src/activity-tracker.ts
  - packages/acn/src/resource-use-gate.ts
  - packages/acn/src/acn-subscriptions.ts
  - packages/acn/src/icn/**
  - packages/acn-protocol/src/schemas/acn-health.ts
---

# ACN service lifecycle

One ACN process owns one authoritative service lifecycle:

```text
Starting(activity, progress?) -> Ready -> Stopping(reason) -> exact exit
```

`Exited` is observed externally through exact process identity. `Installing` is a client
presentation of `Starting`, not an admission mode. Startup or runtime failure enters `Stopping`.

## Process admission

A JIT candidate starts only a stable health/shutdown control server. It remains parent-bound until
it acquires the owner lock and publishes its exact process metadata, and may not construct
application or ICN services before that.

After binding its control endpoint, the candidate acquires `owner-lock.sqlite`, rereads the selected
revision to confirm it is still selected, and publishes `owner.json` with its exact PID, process-start
identity, and port. That publication is process admission. Losing the lock or selection makes the
candidate stop and exit without expensive initialization. Successful admission removes dependence
on its launching manager and permits application startup.

An admitted ACN observes the revision store for a greater selected revision and retires when one
appears. It does not self-revoke on missing, unreadable, or indeterminate coordination state.
Retirement begins only through exact explicit shutdown, idle policy, its own terminal failure, or
process signals. A successor initializes nothing until it owns the lock and the predecessor ACN/ICN
tree is proven absent.

## Readiness and admission

Health, lifecycle observation, application RPC dispatch, root activity admission, and shutdown read
one lifecycle value. The control server exists throughout startup. Application RPC rejects until
the complete application and private ICN exist.

`Ready` installs the RPC application and opens work admission atomically. `Stopping` closes RPC and
activity admission before becoming observable. The first stop reason wins; both transitions are
monotonic and idempotent.

External JIT ensurance treats thirty seconds without authoritative phase change or monotonic
measured progress as a stalled exact `Starting` instance and begins coordinated replacement. ACN
independently owns a five-minute absolute application-startup ceiling that never restarts. Expiry
commits `Stopping(startup-failed)`.

## Activity and idleness

ACN stops after thirty minutes without work or connected-client retention. The first idle period begins only after readiness. A
finite RPC retains demand for its whole handler; shared work continuing after a request retains its
own scoped demand according to [operation ownership](../../architecture/operation-ownership.md).

Health, subscriptions, status/file watches, mirror refetch, display streams, ICN observation,
telemetry, and introspection are non-demand. Explicit renewable client leases retain ACN without
turning observation into demand. The final demand and client-retention release starts the idle timer. Demand
admission and retirement commit are serialized so work cannot enter a service being destroyed.

## Shutdown

Every stop cause uses one process-owned, single-flight shutdown:

```text
commit Stopping and close admission
  -> terminate subscriptions and transports
  -> close application and session scopes
  -> terminate and reap private ICN
  -> exit ACN
  -> external manager proves ACN tree absent and may acquire ownership
```

Caller interruption cannot abandon shutdown. Application-scope closure, notification, fiber,
finalizer, ICN, signal, kill, and reap work are bounded. Abrupt ACN loss closes ICN's private parent
pipe; ICN is not durably recorded or reconciled by the external manager. The ACN never removes or
reassigns its own ACN occurrence.

The control router accepts exact-instance shutdown before application readiness. A matching request
idempotently commits `Stopping` and returns after that transition; a request for another occurrence
cannot stop the process.

## Guarantees

- One lifecycle value governs health, readiness, work admission, idleness, and shutdown.
- No application or ICN work starts before exact owner lock acquisition and metadata publication.
- Losing owner acquisition cannot initialize expensive resources.
- No application work is admitted outside the exact admitted `Ready` ACN.
- Missing or unreadable coordination state cannot make a healthy admitted ACN self-destruct.
- Observation cannot retain ACN, and operation duration cannot replace it.
- Shutdown is single-flight and bounded; the scoped child handle owns exact termination and reap.
