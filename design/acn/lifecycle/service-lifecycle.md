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
durable process state names its exact process and may not construct application or ICN services.

After binding its control endpoint, the candidate atomically changes its exact candidate-owned
revision to `Assigned`. That compare-and-set is process admission. Losing to takeover makes the
candidate stop and exit without expensive initialization. Successful admission removes dependence
on its launching manager and permits application startup.

An assigned ACN does not poll coordination state or self-revoke on missing, unreadable, or newer
state. Retirement begins only through exact explicit shutdown, idle policy, its own terminal failure,
or process signals. Replacement state continues to carry the exact predecessor until an external
manager proves it and its ICN absent.

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

ACN stops after thirty minutes without work. The first idle period begins only after readiness. A
finite RPC retains demand for its whole handler; shared work continuing after a request retains its
own scoped demand according to [operation ownership](../../architecture/operation-ownership.md).

Health, subscriptions, status/file watches, mirror refetch, display streams, ICN observation,
telemetry, and introspection are non-demand. The final demand release starts the idle timer. Demand
admission and retirement commit are serialized so work cannot enter a service being destroyed.

## Shutdown

Every stop cause uses one process-owned, single-flight shutdown:

```text
commit Stopping and close admission
  -> terminate subscriptions and transports
  -> close application and session scopes
  -> terminate and reap private ICN
  -> clear ICN ownership only after exact exit proof
  -> exit ACN
  -> external manager proves ACN exit and advances process state
```

Caller interruption cannot abandon shutdown. Application-scope closure, notification, fiber,
finalizer, ICN, signal, kill, and reap work are bounded. If ACN cannot clear its recorded ICN
ownership, process state retains the exact child for manager reconciliation. The ACN never removes
or reassigns its own ACN occurrence.

The control router accepts exact-instance shutdown before application readiness. A matching request
idempotently commits `Stopping` and returns after that transition; a request for another occurrence
cannot stop the process.

## Guarantees

- One lifecycle value governs health, readiness, work admission, idleness, and shutdown.
- No application or ICN work starts before exact process admission.
- Losing candidate admission cannot initialize expensive resources.
- No application work is admitted outside the exact assigned `Ready` ACN.
- Missing or unreadable process state cannot make a healthy assigned ACN self-destruct.
- Observation cannot retain ACN, and operation duration cannot replace it.
- Shutdown is single-flight, bounded, and retains exact child identity until exit proof.
