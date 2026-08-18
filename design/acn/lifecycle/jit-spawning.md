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
client runtime --ensure(target)--> AcnInstanceManager <--> AcnOwnerStore
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
Changesets changes the CLI version. Development generation increments the machine-local counter
at `~/.magnitude/acn/development-revision-counter` and adds it to that allocation. The
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
defines how `AcnInstance`, `AcnInstanceManager`, `AcnJitRuntime`, and `AcnServiceLifecycle` use that
protocol; it does not define or extend the shared protocol surface.

Schema, statements, decoding, transaction ordering, and typed error translation exist once in the
protocol package. Bun and Node adapters only open scoped connections, execute bound statements,
query rows, close connections, and classify native failures.

`AcnOwnerStore` absorbs expected SQLite contention within one bounded store operation. It returns a
validated ownership snapshot or compare-and-replace outcome, or fails typed when no trustworthy
result can be produced. Managers and ACNs do not retry surfaced store failures.

## Change protocol

A candidate derives its exact process identity and binds health/shutdown on an OS-assigned loopback
port before admission, but starts no application or ICN service. It rereads the expected owner,
proves that predecessor's dedicated process tree absent, and calls `replaceOwner`. Only `Replaced`
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
startup phase, readiness, client leases, or idle retention.

Only after admission may the ACN initialize application and ICN services. Replacement is initiated
by a manager that has observed a lower live revision and prepared its successor; an ACN does not
self-retire from durable version state.

The manager's private exhaustive state projection covers ready, starting, stopping, unavailable,
contradictory health, lower/equal/newer live revision, stale owner, surviving descendant tree,
pending/exited/stalled candidate, and launchable absence. Every state has an explicit action and
fixed deadline. One ensurance occurrence launches at most one candidate and cannot silently turn a
failed launch into a respawn.

Candidate stderr is drained while the process runs and retained only as a bounded tail. Admission
disarms candidate cleanup but does not discard the spawning manager's exit observation. Until
readiness, an exit before or after admission reports its exit code and retained diagnostic instead
of being collapsed into generic coordination loss.

Mutation and final ready adoption reread the complete owner required by their action. Health
revision has authority only while exact process inspection proves that same owner occurrence live.

Policy uses Effect `Duration`, monotonic `Clock`, bounded `Schedule`, and `TestClock`. Initial bounds
are one second polling, two seconds per health request, thirty seconds without an observable health
response, thirty seconds for candidate admission, five minutes absolute application startup while
health remains `Starting` (including long install phases with unchanged optional diagnostics, such as
Resolving, PreparingBackend, and Starting), five seconds for stopping, two seconds after TERM, two
seconds after KILL, and ten minutes absolute per ensurance occurrence. Store contention is bounded
inside each store operation rather than by its consumers. A live `Starting` owner is not retired
merely because its phase or measured progress is unchanged; only the absolute startup ceiling and
loss of health observation authorize retirement during startup. Optional diagnostics never extend
either absolute ceiling.

## Administrative stop

`AcnInstanceManager.stop` observes the current owner, sends shutdown, then reaps the exact process
tree with bounded term-then-kill escalation. It does not kill clients, resolve an artifact, start
an ACN, or directly manage the ACN's private ICN child.

Before shutdown and each signal escalation, the manager rereads the same complete owner and checks
the root identity. A changed owner is not targeted. Root absence does not suppress process-group
signaling: a surviving descendant group is still retired and exact group absence is required before
replacement.

## Guarantees

- Exact owner replacement is the only durable coordination fact.
- No endpoint is projected from a stale, lower-revision, starting, or stopping owner.
- No candidate starts application/ICN before atomic owner admission.
- Two candidates observing the same predecessor cannot both commit.
- A predecessor row is replaced only after exact process-tree absence proof.
- A lower live owner is disrupted only after successor launch material is prepared.
- An older client never replaces an equal or newer live owner.
- Process death removes an owner's revision authority without durable cleanup.
- Every raw child is scope-owned until exact owner publication.
- Every admitted ACN continuously proves that the owner row still names it.
- No stale manager action targets a changed owner.
- Observation uncertainty authorizes neither adoption nor unbounded waiting.
- One ensure cannot turn a failed launch or startup into an implicit retry loop.
- Every ensure and candidate occurrence has one finite terminal result.
- Failure to prove exact tree absence fails typed and never permits overlapping service trees.
