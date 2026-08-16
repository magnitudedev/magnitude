---
applies_to:
  - packages/acn/src/service-lifecycle.ts
  - packages/acn/src/server.ts
  - packages/acn/src/ownership-monitor.ts
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
its exact owner row commits, and may not construct application or ICN services before that.

After binding its control endpoint, the candidate reads the complete current owner, proves that
predecessor's dedicated process tree absent, then atomically replaces the singleton SQLite owner
row only if the complete owner remains unchanged. That commit is process admission. Owner mismatch
makes the candidate stop and exit without expensive initialization. Successful admission removes
dependence on its launching manager, installs the mandatory lifetime owner monitor, and permits
application startup.

An admitted ACN does not poll durable version intent, but it continuously proves that the complete
owner row still equals the row it admitted. A confirmed missing or changed row begins
`Stopping(ownership-lost)`. Any surfaced owner-store failure means ownership can no longer be
proven and fails closed through `Stopping(fatal)`. A manager prepares a successor before asking a
lower-revision live owner to stop, then proves the predecessor ACN/ICN tree absent before the
successor may commit ownership. Retirement otherwise begins through exact explicit shutdown, idle
policy, ownership loss, the ACN's own terminal failure, or process signals.

## Readiness and admission

Health, lifecycle observation, application RPC dispatch, root activity admission, and shutdown read
one lifecycle value. The control server exists throughout startup. Application RPC rejects until
the complete application and private ICN exist.

`Ready` installs the RPC application and opens work admission atomically. `Stopping` closes RPC and
activity admission before becoming observable. The first stop reason wins; both transitions are
monotonic and idempotent.

External JIT ensurance treats observable `Starting` health as live regardless of whether optional
phase or measured-progress diagnostics change. It bounds startup with an absolute five-minute
ceiling and separately bounds loss of observable health. ACN independently owns a five-minute
absolute application-startup ceiling that never restarts. Expiry commits
`Stopping(startup-failed)`.

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
  -> bounded activity drain
  -> terminate subscriptions and transports
  -> close application and session scopes
  -> terminate and reap private ICN
  -> exit ACN
  -> external manager proves ACN tree absent and may acquire ownership
```

`beginStopping` completes immediately after the atomic stopping/admission transition; it never
awaits drain, finalizers, child shutdown, or exit. The server-owned supervisor performs every later
step with a fixed deadline and disconnects cooperative work before applying its timeout, so an
uninterruptible finalizer cannot retain escalation. Abrupt ACN loss closes ICN's private parent pipe;
ICN is not durably recorded or reconciled by the external manager. The ACN never removes or
reassigns its own ACN occurrence.

The control router accepts shutdown before application readiness. The process receiving the request
idempotently commits `Stopping` and returns after that transition. Safety against delayed shutdown
belongs to manager-side owner and exact-process revalidation, not a required endpoint token.

## Guarantees

- One lifecycle value governs health, readiness, work admission, idleness, and shutdown.
- No application or ICN work starts before atomic exact-owner admission.
- Losing owner acquisition cannot initialize expensive resources.
- No application work is admitted outside the exact admitted `Ready` ACN.
- A confirmed missing or changed owner row stops the admitted ACN, and any store failure fails
  closed rather than leaving an unfenced service alive.
- Observation cannot retain ACN, and operation duration cannot replace it.
- The stopping transition is single-flight; cooperative teardown and external exact-tree escalation
  are independently bounded.
