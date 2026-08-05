---
applies_to:
  - packages/acn/src/service-lifecycle.ts
  - packages/acn/src/daemon-lifecycle.ts
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

`Exited` is observed externally through exact process identity; a process cannot publish proof of
its own death.
`Installing` is a client presentation of `Starting`, not an admission mode. Startup or runtime
failure enters `Stopping`; terminal launch/removal failure is reported after exact process outcome.

## Authority and admission

Health, lifecycle observation, RPC dispatch, root activity admission, and shutdown all read the
same lifecycle value. `Ready` installs the complete RPC application and opens admission atomically.
`Stopping` closes RPC and activity admission before it becomes observable; the first stop reason
wins and the transition is monotonic and idempotent.

The stable control server exists throughout startup. Health exposes authoritative startup activity,
while application RPC rejects until `Ready`. Readiness requires the exact canonical grant and
machine-owner epoch, the private ICN, and the complete application layer.

A healthy ACN retires only from explicit shutdown, idle policy, its own terminal failure, or a valid
fenced revocation. Missing, unreadable, malformed, or delayed coordination evidence is uncertainty,
not ownership loss. Revocation and admission share the fence, so no work is accepted after the grant
is superseded.

Startup has two deliberately different liveness bounds. External JIT ensurance treats 30 seconds
without an authoritative phase change or monotonic measured progress as a stalled exact candidate
and begins fenced replacement. Independently, ACN owns a five-minute absolute ceiling from
application startup entry; it never restarts, even when progress changes, and expiry commits
`Stopping(startup-failed)`. The shorter window prevents a hung candidate from blocking clients,
while the process-owned ceiling guarantees terminalization even if every observing client exits.

## Activity and idleness

ACN stops after 30 minutes without work. The first idle period begins only after readiness. A finite
RPC holds a claim for the entire handler; shared work continuing after a request retains its own
scoped claim according to [operation ownership](../../architecture/operation-ownership.md).

Observation does not retain ACN: health, subscriptions, status and file watches, mirror refetch,
display streams, ICN observation, telemetry, and introspection are non-demand. The loopback HTTP
server imposes no idle deadline on an active handler; elapsed response time is never service failure.

The final claim starts the idle timer. Claim acquisition and retirement commit are serialized, so
work cannot enter a generation being destroyed and stale cleanup cannot affect a successor.

## Shutdown

Every stop cause uses one process-owned, single-flight terminalization:

```text
commit Stopping and close admission
  -> atomically detach subscriptions
  -> bounded best-effort terminal delivery and transport close
  -> close application and resident session scopes
  -> terminate and reap private ICN
  -> release exact machine ownership and instance visibility
  -> host observes exact ACN exit
```

ACN shutdown closes session scopes directly; it does not wait on session idle retirement whose work
claims belong to those scopes. Caller interruption cannot abandon shutdown. Notification writes,
fiber interruption, finalizers, cooperative cleanup, ICN shutdown, signal, kill, and reaping are
bounded within an overall teardown limit.

ACN retains machine ownership until normal teardown has reaped ICN. If the cooperative limit
expires, the client holding the fenced JIT replacement claim escalates exact process removal; the
ACN does not release ownership merely because its own cleanup stalled. Timeout causes escalation or
typed failure, never proof of death or ownership theft.

The stable control router accepts an exact fenced shutdown request before application readiness.
The request idempotently commits `Stopping` and returns after that transition; it does not wait for
teardown. A mismatched instance or epoch cannot stop the process.

## Guarantees

- One lifecycle value governs health, readiness, admission, idleness, and shutdown.
- Every nonterminal phase has a live owner capable of terminalizing it.
- No application work is admitted outside the exact fenced `Ready` grant.
- Observation cannot retain ACN; elapsed operation duration cannot replace it.
- Shutdown cannot be retained indefinitely by subscribers, sessions, finalizers, or ICN.
- Ownership is released only after dependent runtime authority has ended.
