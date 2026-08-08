---
applies_to:
  - packages/sdk/src/acn-jit/**
  - packages/acn-protocol/src/acn-identity.ts
  - packages/acn-protocol/src/acn-revision.ts
  - packages/acn-protocol/src/coordination/**
  - packages/acn/src/server.ts
  - packages/acn/src/icn/**
  - packages/version/scripts/generate-version.ts
  - packages/version/scripts/advance-acn-revision.ts
  - packages/version/acn-revision.json
  - desktop/src/main.ts
  - web/scripts/dev-server.ts
---

# JIT ACN instance management and upgrades

Independent hosts sharing one Magnitude data root coordinate to obtain one usable ACN without a
resident coordinator. `AcnInstanceManager` owns complete endpoint acquisition. `AcnRevisionStore`
and `AcnOwnerStore` project the shared facts from one SQLite database; neither is a daemon or a
generic coordination service.

```text
client runtime --ensure(target)--> AcnInstanceManager <--> AcnRevisionStore + AcnOwnerStore
                                                           |
                                                           +--> exact owner ACN
```

Every ensurance occurrence resolves exactly once to an exact `AcnInstance<AcnReady>` or a typed
terminal failure within its absolute deadline. Missing ownership, replacement, delayed startup,
owner death, and candidate launch are bounded intermediate states, never endpoint results.

## Identity and success

ACN version is ACN identity. PID plus process-start identity names one exact process occurrence;
the instance ID is its RPC identity. Revision is one positive safe integer. Development computes a
scalar revision before registration and thereafter follows exactly the same protocol as releases.
Registered revisions are permanent and selection is their maximum.

Each versioned release source allocates one checked-in scalar revision, advanced by one whenever
Changesets changes the CLI version. Development generation increments the machine-local counter
at `~/.magnitude/acn/development-revision-counter` and adds it to that allocation. The
counter is ephemeral build state: it is neither a coordination prerequisite nor part of the
coordination database, and ACN processes observe only the resulting scalar revision.

`AcnInstance<AcnReady>` is the only endpoint result. Projection requires a selected revision, the
complete owner row, an exact live process identity, HTTP `200` ready health whose PID and revision
match those facts, and final rereads confirming the same owner and selection. Readiness is
selection-time evidence; later transport recovery handles retirement.

Each host has a private launch path describing the identities that host can launch and how it
prepares one supported identity. A local development command supports only its exact build
identity; published-release acquisition supports release identities. Commands never cross host
boundaries. Launch preparation is a private dependency of the local manager, not a cross-host
domain capability.

Launch material is prepared before its revision is registered. A manager launches only when its
prepared revision is selected. An older manager may adopt a ready newer owner but never launches a
binary under that newer revision.

## Durable authority

The complete immutable cross-version surface is defined only by
[ACN cross-version coordination protocol](./cross-version-coordination-protocol.md). This document
defines how `AcnInstance`, `AcnInstanceManager`, `AcnJitRuntime`, and `AcnServiceLifecycle` use that
protocol; it does not define or extend the shared protocol surface.

Schema, statements, decoding, transaction ordering, and typed error translation exist once in the
protocol package. Bun and Node adapters only open scoped connections, execute bound statements,
query rows, close connections, and classify native failures.

## Change protocol

A candidate derives its exact process identity and binds health/shutdown on an OS-assigned loopback
port before admission, but starts no application or ICN service. It rereads the expected owner,
proves that predecessor's dedicated process tree absent, and calls `replaceOwner`. Only `Replaced`
is admission; owner or selection mismatch makes the candidate exit.

The candidate stays parent-bound and scope-owned until admission commits. Parent loss and each
atomic admission attempt are serialized by an Effect semaphore; state is an Effect `Ref` and the
one-shot parent-loss signal is a `Deferred`. Retries occur outside that critical section so parent
loss can win between contended attempts. The spawning manager keeps exact candidate cleanup armed
until it observes the owner row equal that candidate and closes the parent channel. Thus every
instant is owned either by manager cleanup or by a complete admitted owner row.

Only after admission may the ACN initialize application and ICN services. It begins retirement only
after positively selecting a different required revision. Indeterminate selection does not retire
a usable owner.

The manager's private exhaustive state projection covers ready, starting, stopping, unavailable,
contradictory health, stale owner, surviving descendant tree, pending/exited/stalled candidate,
newer unsupported selection, and launchable absence. Every state has an explicit action and fixed
deadline. One ensurance occurrence launches at most one candidate and cannot silently turn a failed
launch into a respawn.

Candidate stderr is drained while the process runs and retained only as a bounded tail. A candidate
exit reports its exit code and retained diagnostic instead of being collapsed into generic
coordination loss.

Because selection and owner are separate ordinary reads, observation uses an owner–selection–owner
sandwich. If the complete owner differs, the manager re-observes instead of interpreting a mixed
pair. Mutation and final ready adoption then perform the narrower rereads required by their action.

Policy uses Effect `Duration`, monotonic `Clock`, bounded `Schedule`, and `TestClock`. Initial bounds
are one second polling, two seconds per health request, thirty seconds without an observable health
response, thirty seconds for candidate admission, five minutes absolute application startup while
health remains `Starting` (including long install phases with unchanged optional diagnostics, such as
Resolving, PreparingBackend, and Starting), five seconds for stopping, two seconds after TERM, two
seconds after KILL, and ten minutes absolute per ensurance occurrence. A live `Starting` owner is not
retired merely because its phase or measured progress is unchanged; only the absolute startup
ceiling and loss of health observation authorize retirement during startup. Optional diagnostics
never extend either absolute ceiling.

## Administrative stop

`AcnInstanceManager.stop` observes the current owner, sends shutdown, then reaps the exact process
tree with bounded term-then-kill escalation. It does not kill clients, resolve an artifact, start
an ACN, or directly manage the ACN's private ICN child.

Before shutdown and each signal escalation, the manager rereads the same complete owner and checks
the root identity. A changed owner is not targeted. Root absence does not suppress process-group
signaling: a surviving descendant group is still retired and exact group absence is required before
replacement.

## Guarantees

- Revision selection and owner replacement are the only durable coordination facts.
- Registered and selected revision never regress.
- No endpoint is projected from an unselected, starting, or stopping state.
- No candidate starts application/ICN before atomic owner admission.
- Two candidates observing the same predecessor cannot both commit.
- A predecessor row is replaced only after exact process-tree absence proof.
- Only a host whose target revision is selected invokes its launch path.
- Every raw child is scope-owned until exact owner publication.
- No stale manager action targets a changed owner.
- Observation uncertainty authorizes neither adoption nor unbounded waiting.
- One ensure cannot turn a failed launch or startup into an implicit retry loop.
- Every ensure and candidate occurrence has one finite terminal result.
- Failure to prove exact tree absence fails typed and never permits overlapping service trees.
