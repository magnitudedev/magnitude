---
applies_to:
  - packages/sdk/src/acn-jit/**
  - packages/acn-protocol/src/acn-identity.ts
  - packages/acn-protocol/src/process-state.ts
  - packages/acn/src/server.ts
  - packages/acn/src/icn/**
  - desktop/src/main.ts
  - web/scripts/dev-server.ts
---

# JIT ACN ensurance and upgrades

For one Magnitude data root, independent client hosts coordinate to obtain one usable ACN. There is
no coordinator process. `AcnJitRuntime` owns client policy; `AcnProcessManager` owns local ACN process
reconciliation; one durable `AcnProcessState` serializes independent managers and candidate ACNs.

```text
AcnJitRuntime --ensure(identity)--> AcnProcessManager <--> AcnProcessState
                                           |
                                           +--> exact candidate ACN
```

`ensure(identity)` converges on that identity or a newer usable `Ready` ACN. Missing assignment,
active replacement, delayed startup, manager death, and takeover are intermediate states. Ensurance
ends only with a usable exact instance or a typed condition that prevents safe convergence.

## Identity

ACN version is ACN identity. An ACN instance ID plus PID and process-start identity identifies one
exact process occurrence. No protocol, SDK, target, priority, generation, or coordination identity
competes with ACN version.

Each client starts from its bundled identity and permanently adopts a usable newer `Starting` or
`Ready` ACN. Durable process state also retains a monotonic `identityFloor`. Every launch targets the
higher of the caller identity and that floor, so a client that has not yet observed an upgrade
cannot revive an older ACN.

An equal or lower request joins a sufficient assigned ACN or active change. Artifact resolution is
caller-owned preparation, not authority: after resolution the manager rereads process state and
classifies the request again. It may use the artifact only when its identity still equals the target
the request is authorized to start. A higher request atomically raises an active change target and
identity floor while preserving the change identity. A lower candidate then either loses admission
or, if it already won, is replaced as the exact assigned predecessor.

## ACN process state

The highest complete immutable revision is the complete assignment authority:

```text
AcnProcessState
  revision          compare-and-set fence
  identityFloor     monotonic launch floor
  mode
    Unassigned
    Assigned(exact ACN, optional exact ICN)
    Changing(change identity, purpose, exact owner, phase)
```

`Changing` is owned either by an exact manager process reconciling `RetiringAssigned`,
`RetiringCandidate`, `BlockedCandidateCleanup`, or `Spawning`, or by the exact unadmitted candidate
responsible for admission. Only one change and one owner exist. A blocked cleanup phase retains the
exact unreaped candidate and typed reason; it never authorizes another spawn.

Writers read revision `N`, validate one typed command through the state reducer, and exclusively
publish complete revision `N+1`. Concurrent writers for `N+1` have one winner. The revision is the
only fencing value; there is no launch term, election token, lease, heartbeat, current pointer,
instance inventory, completion marker, machine-owner record, or separate owned-ICN record.

Malformed or unreadable highest state is a typed protocol failure. It is never skipped or treated as
absence. Revisions are not deleted on the correctness path.

## Assignment changes

```text
Unassigned
  -> Changing(manager, Ensure(target), Spawning)
  -> Changing(candidate)
  -> Assigned(candidate)

Assigned(current)
  -> Changing(manager, Ensure(target), RetiringAssigned(current))
  -> Changing(manager, Ensure(target), Spawning)
  -> Changing(candidate)
  -> Assigned(candidate)
```

Beginning replacement does not revoke the predecessor. `RetiringAssigned` continues to identify the
admitted ACN while the manager requests shutdown, escalates against that exact occurrence, and
proves its recorded ICN absent. Only that proof permits `Spawning`. Failed proof returns a typed
blocked result while retaining the unreaped occurrence; it never authorizes another candidate.

After spawn, the manager records the exact candidate and atomically transfers change ownership to
it. Before that transfer the candidate is parent-bound and may not initialize application services.
The candidate binds its stable control endpoint and admits itself by compare-and-setting the exact
candidate-owned revision to `Assigned`. Only successful admission permits application or ICN
startup.

The raw child is a scoped pre-handoff resource. Spawn succeeds only with a PID and immediately
installs bounded stop-and-reap cleanup. The manager then obtains process-start identity and commits
the exact candidate to process state. Only that successful commit permits one atomic handoff that
releases the bootstrap gate and disarms scoped cleanup:

```text
spawn in child scope
  -> failure/cancellation before handoff: scope stops and reaps child
  -> CandidateSpawned CAS
  -> handoff: release bootstrap and disarm scope cleanup
  -> durable process state owns recovery
```

The spawner internally owns the exit observation needed for bounded cleanup and does not pipe child
output into lifecycle outcomes. Its returned child handle exposes only the mandatory PID and this one ownership transfer, not general
process observation or killing policy. After handoff, cleanup addresses the exact occurrence retained in process state. If the
manager dies before handoff, parent-pipe EOF makes the child exit; if handoff fails, scoped cleanup
remains armed.

Admission and takeover contend on the same next revision:

```text
candidate wins -> Assigned; takeover must preserve it
manager wins   -> candidate admission fails; candidate exits without ICN
```

Candidate replacement and candidate failure are distinct transitions. A live stalled or
lower-identity candidate is retired before the manager returns to `Spawning`. A candidate already
proven dead without admission commits `Unassigned(Failed)`; the same failed artifact is not retried
without a new ensure request. Failure to prove candidate exit enters `BlockedCandidateCleanup`.

`terminate(instance)` uses the same state machine with `Terminate` purpose and can affect only the
exact supplied assigned occurrence. It never enumerates or kills peers.

Assignment results—admitted, terminated, or failed before admission—are retained in the current
terminal `Assigned` or `Unassigned` state. A later change may replace that result; callers converge
on current truth rather than querying historical operation outcomes. ACN readiness remains owned by
the ACN service lifecycle.

## Ensurance and timing

```text
usable Ready ACN
  -> adopt newer identity
  -> select exact instance

usable Starting ACN
  -> adopt newer identity
  -> observe exact progress

no usable current ACN
  -> two-second publication grace
  -> processManager.launch(identity)
  -> join or advance current assignment change

Starting stalls
  -> processManager.launch(identity, replace = exact instance)
  -> coordinated replacement
```

| Timing                                | Meaning                                                                                                        | Expiry action                                                                |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 2-second publication grace            | Covers short filesystem and process-scheduling publication gaps. It is not a health deadline.                  | Enter `launch`, which rereads state and joins or atomically begins a change. |
| 30-second change/startup stall        | Bounds continuous observation of one unchanged active owner or one exact `Starting` ACN without real progress. | Contend for takeover or replacement through state compare-and-set.           |
| 5-minute absolute ACN startup ceiling | Bounds ACN-owned application startup even if progress continues.                                               | ACN enters `Stopping(startup-failed)` and cleans its runtime.                |

For an active change, exact owner death permits immediate takeover. A live owner requires thirty
seconds of continuously unchanged authoritative state. A revision, phase, or owner change restarts
observation. Repeated reads and animation do not count as progress.

For an assigned `Starting` ACN, authoritative phase change or monotonic measured progress restarts
its stall window. Application RPC duration is not a coordination clock.

Artifact preparation before `Changing` remains caller-owned and interruptible. The successful
transition to `Changing` and creation of its reconciliation fiber form one uninterruptible admission
step into the scoped local process manager. Caller or remote-stream cancellation after admission
removes only that observer. Host shutdown closes the manager scope; durable owner and phase state
remain for another manager to take over.

## Guarantees

- One state revision is the complete assignment authority.
- One compare-and-set winner may change assignment; stale writers cannot commit.
- Identity floor and active change target never regress.
- A higher target upgrades one active change rather than creating a competing operation.
- Artifact resolution never grants authority and an artifact is launched only for its exact target.
- No candidate is admitted before exact predecessor ACN and admitted ICN cleanup.
- No unadmitted candidate may start application or ICN work.
- Every spawned child is scope-owned until its exact durable handoff; every pre-handoff exit path
  stops and reaps it.
- Candidate admission and takeover cannot both win.
- Dead and stalled owners are recoverable without unfenced spawn.
- Candidate failure cannot become an implicit unbounded retry loop.
- Observation uncertainty and elapsed application work do not authorize mutation.
- Repeated startup failure cannot accumulate admitted ACNs or CUDA-owning ICNs.

The unavoidable raw `spawn`-to-publication interval contains no expensive work. A literal guarantee
that a second kernel process can never briefly exist would require a resident supervisor, which is
not part of this design.
