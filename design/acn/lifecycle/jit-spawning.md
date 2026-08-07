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
resident coordinator. `AcnInstanceManager` owns complete endpoint acquisition. Two durable
file-and-lock resources — `AcnRevisionStore` and `AcnOwnerLock` — serialize revision selection and
single-owner publication.

```text
client runtime --ensure(target)--> AcnInstanceManager <--> AcnRevisionStore + AcnOwnerLock
                                                           |
                                                           +--> exact owner ACN
```

Missing owner, replacement, delayed startup, owner death, and candidate launch are intermediate.
Ensurance returns only exact `AcnInstance<AcnReady>` or a typed condition that prevents safe
convergence. There is no public current observation, caller-selected replacement, or ordinary
exact-terminate operation. Administrative stop is an operation of the same `AcnInstanceManager`.

## Identity and success

ACN version is ACN identity. Instance ID plus PID and process-start identity names one exact
occurrence. Revision is a scalar positive safe integer. Published revisions sit on million-wide
band boundaries in `packages/version/acn-revision.json`; development builds claim unique offsets
within the band via filesystem link races during version generation. The selected revision is the
greatest scalar revision among published markers and development markers with at least one active
holder.

`AcnInstance<AcnReady>` is the only endpoint result. Its constructor proves stable ownership,
exact live process, matching ID/identity/PID `Ready` health, and a final stable-state reread for
the same occurrence and endpoint. Readiness is selection-time evidence; later transport recovery
handles retirement after selection.

Each host has a private launch path describing the identities that host can launch and how it
prepares one supported identity. A local development command supports only its exact build
identity; published-release acquisition supports release identities. Commands never cross host
boundaries. Launch preparation is a private dependency of the local manager, not a cross-host
domain capability.

Preparation is part of the authoritative selection protocol. Only a host whose target revision is
selected may prepare and launch. Other clients observe that selection and never attempt to prepare
the owner's target. Preparation is interruptible before acquisition, but cancellation of the
caller cannot abandon an acquired owner lock.

## Durable authority

```text
AcnRevisionStore                        AcnOwnerLock
  D/acn/revisions/<revision>              D/acn/owner-lock.sqlite (SQLite mutex)
    published: zero bytes                   D/acn/owner.json (published while holding)
    development: 16-hex-byte key          Unlocked | Publishing | Locked(owner)
  D/acn/development-holds/<revision>/
    <uuid>.sqlite (SQLite holder mutexes)
```

The selected revision is the entire authority. Published markers are permanent; development
markers participate only while at least one holder mutex is active. Any failure to enumerate,
validate, or probe relevant state makes selection indeterminate; indeterminate selection authorizes
neither admission nor retirement.

The owner lock is a SQLite `BEGIN IMMEDIATE` transaction. While holding it, the owner atomically
publishes `owner.json` with its exact PID, process-start identity, and bound port. The transaction
proves ownership; metadata never does. Rollback, connection close, or process exit releases
ownership through SQLite's OS locks. Unlocked metadata remains predecessor-cleanup evidence until a
successor proves that exact tree absent and atomically replaces it with its own publication.

## Change protocol

```text
no selection -> register/hold revision -> selected
selected, no owner -> acquire owner lock -> publish metadata -> serve
selected, owner exists -> observe health -> adopt or wait
selected, greater revision appears -> retire (replacement)
```

A candidate prepares its revision marker before attempting to acquire the owner lock, may serve
only while holding the owner SQLite transaction, and must reread selection after acquisition. A
non-selected candidate releases the lock and exits without serving.

An owner writes its metadata before exposing HTTP or initializing application/ICN services, retains
the owner transaction through complete service and child teardown, and retires only after positively
observing a greater revision. Observation uncertainty causes no retirement.

Replacement first shuts down and proves absence of the exact predecessor process tree. The
predecessor's private ICN is owned entirely by its ACN scope and exits during orderly scope
finalization or when abrupt ACN loss closes its parent pipe. Failed exit proof retains the
occurrence and fails the change.

A spawned child is scoped and parent-bound until its exact owner is published. The candidate binds
its control endpoint, verifies exact ownership, and publishes before application or ICN startup.
Owner acquisition and health verification contend for the same lock, so two candidates cannot both
serve.

One public ensure binds every observed change by exact revision and resolves its result from the
current selection state even if a later change has already begun. A failed launch is never silently
reclassified as another attempt. After a successful adoption, the ensure may explicitly follow a
causally later sufficient replacement; this is upgrade following, not a retry. A selected owner
satisfying the caller's own minimum revision also remains usable after a higher-target preparation
failure. These rules prevent implicit respawn loops without turning coherent upgrades into failures.

## Reconciliation ownership

Successful owner lock acquisition plus forking its supervisor is one uninterruptible admission
step. One host-local supervisor at a time drives the exact admitted owner; a higher revision
selected within that change discards obsolete preparation before the same supervisor reclassifies
it. Owner publication transfers reconciliation to the owner, so the manager supervisor exits at
handoff. It also exits on foreign owner ownership and never takes the change back. Observer
cancellation does not abandon admitted manager work.

Foreground ensure observers classify and wait. They do not prepare, duplicate cleanup, or spawn.
Exact owner absence or continuous unusable health permits coordinated replacement. Incapable
followers remain observers and cannot turn their local incapability into global change failure.

Policy deadlines use Effect `Duration`, monotonic `Clock`, and `TestClock`. The convergence
durations are the coordination poll interval, the tree reap waits (term then kill), and the
ACN-owned five-minute absolute application-startup ceiling. There is no publication grace: the
owner lock is held before publish, and contenders coordinate through SQLite busy semantics.

## Administrative stop

`AcnInstanceManager.stop` observes the current owner, sends shutdown, then reaps the exact process
tree with bounded term-then-kill escalation. It does not kill clients, resolve an artifact, start
an ACN, or directly manage the ACN's private ICN child.

## Guarantees

- The selected revision is the complete assignment authority and stale writers cannot commit.
- Active revision never regresses.
- No endpoint is projected from an unselected, starting, or stopping state.
- No candidate starts application/ICN before exact owner publication.
- No predecessor retirement begins before a greater revision is selected.
- Only a host whose target revision is selected invokes its launch path.
- A follower-local unsupported target or acquisition failure cannot fail another owner's change.
- Every raw child is scope-owned until exact owner publication.
- Two candidates cannot both serve the same revision.
- Cancellation cannot abandon admitted reconciliation.
- Foreign ownership cannot be taken back by a stale healthy supervisor.
- Observation uncertainty alone never authorizes immediate mutation.
- One ensure cannot turn a failed launch or startup into an implicit retry loop.
- Every launch and startup occurrence is resolved exactly before a sufficient successor is followed.
