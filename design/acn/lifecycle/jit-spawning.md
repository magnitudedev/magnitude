---
applies_to:
  - packages/sdk/src/acn-jit/**
  - packages/acn/src/daemon-registration.ts
  - packages/acn/src/machine-ownership.ts
  - packages/acn/src/peer-acn.ts
  - packages/acn/src/process-identity.ts
  - packages/acn/src/daemon-lifecycle.ts
  - packages/acn-protocol/src/acn-registry.ts
  - packages/acn-protocol/src/schemas/acn-health.ts
  - desktop/src/main.ts
  - web/scripts/dev-server.ts
---

# JIT ACN spawning and handoff

For one Magnitude data root, one host coordination authority selects the ACN allowed to admit work.
A client requests an endpoint satisfying explicit requirements; it never receives direct permission
to spawn, replace, or kill a process.

Opening a compatible newer client must succeed independently of older open clients. Older clients
may join the newer ACN when RPC-compatible; their continuity is best effort, but they cannot
obstruct, displace, or downgrade it. Incompatible active authority is a typed local conflict unless
isolation or explicit forced replacement is authorized.

“Independently” means an older client's recovery or downgrade attempt cannot fail the newer launch.
Already-admitted incumbent work may still delay the cooperative handoff required to preserve it.

## Identities and authority

Every requirement and candidate separates:

| Identity | Meaning |
| --- | --- |
| Coordination protocol | Compatibility of the stable host-coordination envelope |
| RPC protocol | Whether a client may use the ACN |
| Storage protocol | Whether processes may safely share the data root |
| Release priority | Deterministic preference among otherwise compatible candidates |

Human version and logs are diagnostics. Build time is not compatibility or precedence; development
artifacts use stable content identity. The exact artifact, protocols, priority, and process identity
are bound before spawn and cannot be rederived from mutable source state.

The coordinator owns endpoint selection, candidate admission and supervision, fencing epochs,
compatibility, deterministic priority, cooperative handoff, rollback, reconciliation, and exact
removal. CLI processes may instantiate the service independently, but all mutations reduce through
one process-safe durable state machine. Desktop main and web host expose the same authority remotely.

## Coordination state

```text
instances/<instance-id>   immutable process advertisement
coordination              Empty | Active | Handoff, with monotonic epoch
coordination-lock         short exact-process transaction ownership
machine-owner             active-runtime grant carrying instance and epoch
```

An advertisement contains opaque instance ID, bound protocols and priority, endpoint, PID, and
process-start identity. PID alone never authorizes signaling or deletion. Malformed or proven-dead
records are collected; live records are never discarded by age.

Only the coordinator writes authoritative state. A candidate advertises itself and reports its
[service lifecycle](./service-lifecycle.md); it never appoints itself canonical.
`Active(epoch, instance)` and machine ownership name
the same grant. Every health response, RPC request, admission decision, handoff acknowledgement,
shutdown request, and force authorization carries `{ instanceId, epoch }`.

An obsolete process therefore cannot admit work after revocation, and a stale proxy or reused port
cannot satisfy a request for another grant. Missing, malformed, unreadable, or timed-out state never
fabricates absence, a new epoch, or revocation.

The transaction lock is recovered only after exact owner-death proof using PID plus process-start
identity. Long work commits a durable operation phase and releases the transaction lock; elapsed
time never steals live admission ownership.

## Ensure and spawn admission

Discovery is observation. `ensure(requirement)` is host-owned shared work whose result is a
satisfying canonical endpoint, not the survival of one candidate child.

| Revalidated authority | Required result |
| --- | --- |
| Compatible equal/higher `Ready` | Return it |
| Compatible equal/higher `Starting` or admitted candidate | Join and observe it |
| Lower-priority compatible authority | Admit one eligible successor |
| Incompatible active authority | Typed conflict; no automatic eviction |
| Authoritatively `Empty` | Admit one candidate |
| `Handoff` | Join/reconcile that exact operation |
| Unknown or inconsistent evidence | Reconcile or fail locally; do not spawn |

Transport failure, request latency, health timeout, and caller impatience never authorize spawn.
`Starting` is positive evidence of owned startup; a waiter may fail locally but cannot fall through
to an equal or lower candidate.

Candidate admission is outcome-total:

1. Revalidate authority and requirements inside the coordination transaction.
2. Return a satisfying authority, join equivalent work, or reject/defer conflict.
3. Commit host-owned operation identity before spawning.
4. Spawn and publish the exact immutable advertisement.
5. Validate actual artifact, protocols, process identity, and endpoint.
6. Prepare the candidate to standby readiness without canonical publication or peer destruction.
7. Enter handoff only if it remains the deterministic eligible winner.
8. Complete, roll back, or terminalize on every Effect `Exit`.

Standby readiness proves the bound candidate, control endpoint, protocols, and startup dependencies
needed to make handoff recoverable; it is not ACN `Ready` and cannot admit application RPC.

Caller cancellation removes one waiter only. Host failure leaves a durable phase that another
coordinator reconciles. Candidate exit is attempt evidence: if another compatible authority wins,
ensure succeeds with it. Exit becomes candidate failure only when no acceptable authority remains.

## Safe handoff

Automatic replacement is cooperative and may commit only at a quiescent boundary; merely opening a
client is not force authorization.

```text
candidate StandbyReady
  -> commit Handoff(Preparing)
  -> incumbent closes fenced admission and drains owned work
  -> incumbent reports Quiescent and releases machine ownership
  -> candidate acquires the next epoch and reaches Ready
  -> commit Active(candidate)
  -> retire incumbent
```

Before final commit, candidate failure restores the incumbent's admission and `Active` grant.
Incumbent failure is reconciled from exact exit evidence. After commit, the old epoch cannot admit
work. Explicit forced replacement is a separate user/administrative mutation with an exact fenced
target; automatic upgrade never escalates to force while admitted product work remains.

## Exact removal

Only a committed handoff or explicit force operation may remove another instance:

```text
validate instance + epoch + process-start identity
  -> request cooperative shutdown
  -> bounded exact-exit wait
  -> revalidate and terminate
  -> bounded wait
  -> revalidate, force kill, and reap/prove death
  -> remove only exact instance and role records
```

A retained child handle is the strongest authority for a locally spawned candidate; discovered
processes require identity validation before every signal. Timeout escalates removal or returns a
typed failure. It never proves death, permits ownership reuse, or lets delayed cleanup remove a
successor.

## Guarantees and verification

- Exactly one fenced epoch admits work for a data root.
- Selection, machine ownership, and destructive authority cannot disagree.
- Priority never moves backward because of scheduling or last-writer order.
- Compatibility is explicit and independent of priority.
- Unknown observation and client transport failure cannot mutate global ownership.
- Automatic replacement cannot interrupt admitted product work.
- Every durable nonterminal phase has a live owner or deterministic reconciler.
- Caller/host loss cannot orphan launch or handoff.
- A candidate failure before commit preserves or restores the incumbent.
- Generic recovery cannot replay an unsafe mutation.

Conformance uses independent CLI, desktop, and web-host coordinators; equal, older, newer, and
incompatible candidates in every ordering; slow `Starting`; caller and host failure at every phase;
malformed/unreadable coordination; PID reuse; active turns and model work during handoff; and
ambiguous mutation responses. Global invariants are asserted after every transition, not only the
final result.
