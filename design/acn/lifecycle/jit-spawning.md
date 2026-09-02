---
applies_to:
  - packages/sdk/src/acn-jit/**
  - packages/acn-protocol/src/acn-identity.ts
  - packages/acn-protocol/src/acn-revision.ts
  - packages/acn-protocol/src/coordination/**
  - packages/acn/src/server.ts
  - packages/acn/src/ownership-monitor.ts
  - packages/acn/src/icn/**
  - packages/version/scripts/generate-version.ts
  - packages/version/scripts/advance-acn-revision.ts
  - packages/version/acn-revision.json
  - desktop/src/main.ts
  - web/scripts/dev-server.ts
---

# JIT ACN instance management and upgrades

Independent hosts sharing one Magnitude data root coordinate to obtain one usable ACN without a
resident coordinator. `AcnInstanceManager` owns complete endpoint acquisition. `AcnOwnerStore`
projects the one shared fact from SQLite; it is neither a daemon nor a generic coordination
service.

```text
ACN connection --ensure(target)--> AcnInstanceManager <--> AcnOwnerStore
                                                           |
                                                           +--> exact live owner ACN
```

Every ensurance occurrence resolves exactly once to an exact `AcnInstance<AcnReady>` or a typed
terminal failure within its absolute deadline. Missing ownership, replacement, delayed startup,
owner death, and candidate launch are bounded intermediate states, never endpoint results.

## Identity and success

ACN version is ACN identity. PID plus process-start identity names one exact process occurrence;
the instance ID is its RPC identity. Revision is one positive safe integer reported by each live
owner and used to order it against a client's target.

Each versioned release source allocates one checked-in scalar revision, advanced by one whenever
Changesets changes the CLI version. Development generation adds a development counter to that
allocation. It uses an explicitly configured non-negative counter when present; otherwise it
increments the machine-local counter at `~/.magnitude/acn/development-revision-counter`. The
counter is ephemeral build state. ACN processes observe only the resulting scalar revision; no
revision is persisted as coordination authority.

`AcnInstance<AcnReady>` is the only endpoint result. Projection requires the complete owner row, an
exact live process identity, HTTP `200` ready health whose PID matches the owner and whose revision
meets the client's target, and final rereads confirming the same owner and process occurrence.
Readiness is selection-time evidence; later transport recovery handles retirement.

Each host has a private launch path describing the identities that host can launch and how it
prepares one supported identity. A local development command supports only its exact build
identity; published-release acquisition supports release identities. Commands never cross host
boundaries. Launch preparation is a private dependency of the local manager, not a cross-host
domain capability.

Launch material is prepared before a lower live owner is disrupted. An older manager adopts a ready
newer owner but never launches a binary under that newer revision.

## Durable authority

The complete immutable cross-version surface is defined only by
[ACN cross-version coordination protocol](./cross-version-coordination-protocol.md). This document
defines how `AcnInstance`, `AcnInstanceManager`, `AcnConnection`, and `AcnServiceLifecycle` use that
protocol; it does not define or extend the shared protocol surface.

Schema, statements, decoding, transaction ordering, and typed error translation exist once in the
protocol package. Bun and Node adapters only open scoped connections, execute bound statements,
query rows, close connections, and classify native failures.

`AcnOwnerStore` absorbs expected SQLite contention within one bounded store operation. It returns a
validated ownership snapshot or compare-and-replace outcome, or fails typed when no trustworthy
result can be produced. Managers and ACNs do not retry surfaced store failures.

## Lifecycle authority boundaries

The OS boundary is generic. `ExactProcess` identifies one process occurrence, `ProcessGroup`
identifies the group led by that occurrence, and one `ProcessGroupController` inspects what
occupies a pid, observes a group's state (leader live, leader replaced, survivors only, absent),
waits for group exit, and stops a group with identity-checked TERM → KILL → absence-proof
escalation. This adapter contains no ACN owner, revision, health, admission, or convergence policy.

The SDK composes six narrow authorities:

- `AcnOwnerObserver` reads owner, exact-process, and health facts without mutation.
- `AcnConvergenceDecider` is a pure total function from one observation snapshot to one action.
- `AcnDaemonShutdownSupervisor` exclusively owns revalidation and graceful/TERM/KILL control of one
  exact existing daemon process group.
- `AcnCandidateLaunchSupervisor` owns one scoped candidate from spawn through admission or cleanup.
- `AcnDaemonLaunchCommandResolver` resolves a launch command for a supported ACN target without
  starting or stopping processes.
- `AcnEnsuranceCoordinator` executes decisions but owns none of those underlying policies.

The candidate supervisor declares its transition graph with `FSM.defineFSM` — the live states
`NotLaunched → Spawned → Admitted → Ready` plus one terminal `Failed` state that carries the typed
candidate failure. The shutdown supervisor is a serialized linear protocol, not a state machine: one
semaphore serializes each complete shutdown occurrence, and `ProcessGroupController.stop` performs
TERM → KILL → absence proof for daemon shutdown and candidate cleanup alike. Candidate cleanup is scoped and remains armed until durable admission is observed. Before
exact identity confirmation it may target only the raw bootstrap handle; after confirmation it may
target only the group led by that exact process occurrence. Known pre-admission terminal paths
explicitly join typed cleanup, while the scope finalizer is interruption protection and reports
rather than defects on cleanup failure. Shutdown control failures remain in the Effect error
channel as one `AcnDaemonShutdownFailed` wrapper whose `failure` member is the typed union of
underlying store/observation/signal causes.

Failure identity is a tagged type per distinct mechanism; deterministic context — which owner,
which shutdown reason, which signal — travels as structured fields or a typed nesting wrapper,
never as prose reasons, pseudo-tag codes, or flattened context × mechanism class products.
Candidate failures are one taxonomy used both as the `Failed` state payload and as ensure errors;
the decider passes them through as the single `FailCandidate` decision. Invariant violations the
composition makes unreachable (double admission, ready before admission, relaunch) are defects,
not typed errors.

## Change protocol

A candidate derives its exact process identity and binds health/shutdown on an OS-assigned loopback
port before admission, but starts no application or ICN service. It rereads the expected owner,
proves that predecessor's dedicated process group is absent, and calls `replaceOwner`. Only `Replaced`
is admission; owner mismatch makes the candidate exit.

The candidate stays parent-bound and scope-owned until admission commits. Parent loss and each
atomic admission attempt are serialized by an Effect semaphore; state is an Effect `Ref` and the
one-shot parent-loss signal is a `Deferred`. The bounded store operation is raced against parent
loss, while its short atomic transaction remains uninterruptible. The spawning manager keeps exact
candidate cleanup armed until it observes the owner row equal that candidate and closes the parent
channel. Thus every instant is owned either by manager cleanup or by a complete admitted owner row.

After admission, the ACN installs a mandatory scoped monitor before application initialization.
The monitor continuously compares the complete current owner row with the exact row it admitted.
A confirmed missing or changed owner begins `ownership-lost` shutdown; any surfaced store failure
fails closed as fatal shutdown. The monitor runs for the complete admitted lifetime, independent of
startup phase, readiness, client presence, or idle retention.

Only after admission may the ACN initialize application and ICN services. Replacement is initiated
by a manager that has observed a lower live revision and prepared its successor; an ACN does not
self-retire from durable version state.

The pure convergence decider's exhaustive state projection covers ready, starting, stopping, unavailable,
contradictory health, lower/equal/newer live revision, stale owner, surviving process group,
pending/exited/stalled candidate, and launchable absence. Every state has an explicit action and
fixed deadline. One ensurance occurrence launches at most one candidate and cannot silently turn a
failed launch into a respawn.

Candidate stderr is drained by a best-effort Effect fiber while the process runs and retained only as
a bounded tail. Candidate exit observation is governed by the root process's OS exit callback, never
by pipe EOF: descendants may inherit the descriptor without delaying failure observation or client
scope shutdown. Root exit requests diagnostic-capture interruption and snapshots the retained tail.
Admission disarms ordinary scoped
candidate cleanup but does not discard the spawning manager's exit observation or exact group
identity. If an admitted candidate exits or loses its owner row before readiness, the manager stops
and proves absence of that exact process group before it publishes the terminal failure. Until
readiness, an exit before or after admission reports its exit
code and retained diagnostic instead of being collapsed into generic coordination loss.

Mutation and final ready adoption reread the complete owner required by their action. Health
revision has authority only while exact process inspection proves that same owner occurrence live.

Policy uses Effect `Duration`, monotonic `Clock`, bounded `Schedule`, and `TestClock`. Initial bounds
are one second polling, two seconds per health request, exactly one independent confirmation after
an inconclusive health request, thirty seconds without an observable health
response, thirty seconds for candidate admission, five minutes absolute application startup while
health remains `Starting` (including phases with unchanged optional diagnostics, such as Resolving,
PreparingBackend and Starting), five seconds for stopping, two seconds after TERM, two
seconds after KILL, and ten minutes absolute per ensurance occurrence. Store contention is bounded
inside each store operation rather than by its consumers. A live `Starting` owner is not retired
merely because its phase or measured progress is unchanged; only the absolute startup ceiling and
loss of health observation authorize retirement during startup. Optional diagnostics never extend
either absolute ceiling.

## Administrative stop

`AcnInstanceManager.stop` observes the current owner and delegates the complete bounded graceful,
terminate, kill, and absence-proof protocol to `AcnDaemonShutdownSupervisor`. It does not kill
clients, resolve an artifact, start an ACN, or directly manage the ACN's private ICN child.

The supervisor rereads the same complete owner and checks the root identity before the graceful
attempt and again before signal escalation; within escalation, every signal delivery itself
revalidates exact leader identity and refuses a changed occurrence. A changed owner or reused PID
is never targeted. Root absence does not suppress process-group signaling: a surviving descendant
group is still retired and exact group absence is required before replacement.

## Guarantees

- Exact owner replacement is the only durable coordination fact.
- No endpoint is projected from a stale, lower-revision, starting, or stopping owner.
- No candidate starts application/ICN before atomic owner admission.
- Two candidates observing the same predecessor cannot both commit.
- A predecessor row is replaced only after exact process-group absence proof.
- A lower live owner is disrupted only after successor launch material is prepared.
- An older client never replaces an equal or newer live owner.
- Process death removes an owner's revision authority without durable cleanup.
- Every raw child is scope-owned until exact owner publication.
- Candidate cleanup targets only the raw bootstrap handle before identity confirmation and only the
  confirmed exact process group afterward.
- Every known pre-admission terminal path joins cleanup; finalizer cleanup failure is reported and
  never becomes an unchecked defect.
- Every admitted ACN continuously proves that the owner row still names it.
- An admitted candidate that exits or loses ownership before readiness has its exact process group
  retired before failure is published.
- No stale manager action targets a changed owner.
- Existing-daemon mutation exists only inside `AcnDaemonShutdownSupervisor`.
- Shutdown-control failures remain typed in the supervisor Effect error channel.
- Candidate process ownership exists only inside `AcnCandidateLaunchSupervisor`.
- Every candidate transition is admitted by its declared FSM graph; every shutdown occurrence is
  serialized end to end.
- Observation uncertainty authorizes neither adoption nor unbounded waiting.
- One ensure cannot turn a failed launch or startup into an implicit retry loop.
- Every ensure and candidate occurrence has one finite terminal result.
- Failure to prove exact process-group absence fails typed and never permits overlapping service groups.
