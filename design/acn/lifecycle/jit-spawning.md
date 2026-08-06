---
applies_to:
  - packages/sdk/src/acn-jit/**
  - packages/acn-protocol/src/acn-identity.ts
  - packages/acn-protocol/src/process-state.ts
  - packages/acn/src/server.ts
  - packages/acn/src/icn/**
  - packages/version/scripts/generate-version.ts
  - desktop/src/main.ts
  - web/scripts/dev-server.ts
---

# JIT ACN ensurance and upgrades

Independent hosts sharing one Magnitude data root coordinate to obtain one usable ACN without a
resident coordinator. `AcnEnsurer` owns complete endpoint acquisition and one durable
`AcnProcessState` serializes managers and candidate ACNs.

```text
client runtime --ensure(minimum identity)--> AcnEnsurer <--> AcnProcessState
                                                        |
                                                        +--> exact candidate ACN
```

Missing assignment, replacement, delayed startup, manager death, and candidate admission are
intermediate. Ensurance returns only exact `ReadyAcn` or a typed condition that prevents safe
convergence. There is no public current observation, caller-selected replacement, or ordinary
exact-terminate operation. Administrative stop is the separate local `AcnDaemonAdministrator`.

## Identity and success

ACN version is ACN identity. Instance ID plus PID and process-start identity names one exact
occurrence. Every classification targets the higher of caller minimum identity and the durable
monotonic `identityFloor`.

`ReadyAcn` is the only endpoint result. Its constructor proves stable `Assigned`, exact live
process, matching ID/identity/PID `Ready` health, and a final stable-state reread for the same
occurrence and endpoint. Revision-only ICN record changes do not invalidate the occurrence.
Readiness is selection-time evidence; later transport recovery handles retirement after selection.

Each host has an `AcnLaunchSource` describing the identities that host can launch and how it prepares
one supported identity. A local development command supports only its exact build identity;
published-release acquisition supports release identities. Commands never cross host boundaries.

Preparation is part of the authoritative change protocol. Only the exact manager owning
`Preparing` may prepare the target. Other clients observe that change and never attempt to prepare
the owner's target. Preparation is interruptible before admission, but cancellation of the caller
cannot abandon an admitted `Preparing` change.

## Durable authority

```text
AcnProcessState
  revision          compare-and-set fence
  identityFloor     monotonic launch floor
  mode
    Unassigned
    Assigned(exact ACN, optional exact ICN)
    Changing(change revision, purpose, exact owner, phase)
```

The highest complete immutable consecutive revision is the entire authority. Writers validate one
typed reducer command and exclusively publish the next revision. There is no current pointer,
launch lease, second owned-ICN record, manager registry, or completion store. Invalid or unreadable
highest state fails typed and is never treated as absence.

`Changing` has one `Ensure` or `Terminate` purpose and one exact manager or candidate owner. Its
`changeRevision` is the exact change identity used by every observer. Manager phases retain the
exact predecessor/candidate until cleanup proof. Blocked cleanup never authorizes another spawn.

## Change protocol

```text
Unassigned
  -> Changing(manager, Ensure, Preparing(no incumbent))
  -> Changing(manager, Ensure, Spawning)
  -> Changing(candidate)
  -> Assigned(candidate)

Assigned(current)
  -> Changing(manager, Ensure, Preparing(current))
  -> Changing(manager, Ensure, RetiringAssigned(current))
  -> Changing(manager, Ensure, Spawning)
  -> Changing(candidate)
  -> Assigned(candidate)
```

Preparation success advances atomically to retirement or spawning. Preparation failure restores
the retained incumbent, or `Unassigned` when there was none, and records failure for that exact
change. The identity floor remains a launch fence, but does not make a retained incumbent unusable
to a client whose own minimum identity it satisfies.

Replacement first shuts down and proves absence of the exact predecessor and its recorded ICN.
The ACN normally stops its child ICN; recorded ICN cleanup is the orphan backstop. Failed proof
retains the occurrence and fails the change.

A spawned child is scoped and parent-bound until its exact candidate is committed. Only successful
`CandidateSpawned` permits atomic handoff, releasing bootstrap and disarming cleanup. The candidate
binds its control endpoint, verifies exact ownership, and compare-and-sets `CandidateAdmitted`
before application or ICN startup. Admission and takeover contend for the same next revision, so
both cannot win.

One public ensure binds every observed change by exact `changeRevision` and resolves its result from
immutable revision history even if a later change has already begun. A failed change is never
silently reclassified as another attempt. After a successful admission, the ensure may explicitly
follow a causally later sufficient replacement; this is upgrade following, not a retry. A retained
incumbent satisfying the caller's own minimum also remains usable after a higher-target preparation
failure. These rules prevent implicit respawn loops without turning coherent upgrades into failures.

## Reconciliation ownership

Successful change compare-and-set plus forking its supervisor is one uninterruptible admission
step. One host-local supervisor at a time drives the exact admitted change; a higher target committed
within that change discards obsolete preparation before the same supervisor reclassifies it.
`CandidateSpawned` transfers reconciliation to the candidate, so the manager supervisor exits at
handoff. It also exits on foreign manager ownership and never takes the change back. Observer
cancellation does not abandon admitted manager work.

Foreground ensure observers classify and wait. They do not prepare, duplicate cleanup, or spawn.
Exact owner absence or 30 seconds of one unchanged owner/revision permits fenced takeover by a host
whose launch source supports the target. Takeover returns non-transferable preparation to
`Preparing`; the next revision fences the old owner. Incapable followers remain observers and
cannot turn their local incapability into global change failure.

For a stable exact live assignment, transient health transport/schema/identity failure is internal
evidence. Meaningful `Starting` progress resets the 30-second deadline. Initial `Stopping`, exact
death, or continuous unusable health permits coordinated replacement. Once an ensure has joined a
startup occurrence, its stop/death/stall is terminal for that ensure rather than another retry.

Policy deadlines use Effect `Duration`, monotonic `Clock`, and `TestClock`. The only convergence
durations are the 30-second unchanged owner/assigned-startup stall and the ACN-owned five-minute
absolute application-startup ceiling. There is no publication grace: `Changing` is committed before
spawn, and contenders coordinate through next-revision compare-and-set.

## Administrative stop

`stopCurrent` converts stable assignment or an active Ensure change to durable `Terminate`, then
joins or takes over exact cleanup until stable `Unassigned`. It does not kill clients, resolve an
artifact, or start an ACN. Stopping the ACN owns normal ICN shutdown; exact recorded ICN termination
is only the orphan fallback.

## Guarantees

- One revision is the complete assignment authority and stale writers cannot commit.
- Identity floor and active target never regress.
- No endpoint is projected from `Unassigned`, `Changing`, `Starting`, or `Stopping`.
- No candidate starts application/ICN before exact predecessor cleanup and candidate admission.
- No predecessor retirement begins before successful target preparation.
- Only the exact preparation owner invokes its launch source.
- A follower-local unsupported target or acquisition failure cannot fail another owner's change.
- Every raw child is scope-owned until exact durable handoff.
- Candidate admission and takeover cannot both win.
- Cancellation cannot abandon admitted reconciliation.
- Foreign ownership cannot be taken back by a stale healthy supervisor.
- Observation uncertainty alone never authorizes immediate mutation.
- One ensure cannot turn a failed change or startup into an implicit retry loop.
- Every change and startup occurrence is resolved exactly before a sufficient successor is followed.
