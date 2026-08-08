---
applies_to:
  - packages/acn-protocol/src/coordination/**
  - packages/acn-protocol/src/schemas/acn-health.ts
  - packages/sdk/src/acn-jit/local-acn-instance-manager.ts
  - packages/acn/src/server.ts
  - packages/acn/src/binary.ts
  - packages/acn/src/version.ts
  - packages/sdk/src/version.ts
  - packages/version/scripts/generate-version.ts
---

# ACN cross-version coordination protocol

This document is the complete required surface shared by ACN and client versions. No path, field,
encoding, SQL operation, HTTP behavior, or process expectation in this document may change, and no
additional behavior may become required for coordination or convergence, without explicit approval.
Nothing outside this document is a cross-version coordination prerequisite.

## Filesystem and database surface

For Magnitude data root `D`, the complete shared filesystem surface is:

```text
D/acn/coordination.sqlite
```

The database contains exactly one semantic table:

```sql
CREATE TABLE owner (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  pid INTEGER NOT NULL CHECK (pid > 0 AND pid <= 9007199254740991),
  process_start_identity TEXT NOT NULL
    CHECK (length(process_start_identity) > 0),
  port INTEGER NOT NULL CHECK (port BETWEEN 1 AND 65535)
);
```

There is no persisted revision election, desired version, process history, workflow, lifecycle,
generation, lease, or compatibility state. Every connection uses SQLite rollback-journal mode and
`busy_timeout=0`. `SQLITE_BUSY` is the only contention result and callers retry it only within a
fixed operation deadline. Connections and transactions are short-lived and never represent process
ownership.

## Owner value and authority

`owner` contains zero or one row. Its complete semantic value is:

```json
{"pid":1234,"processStartIdentity":"<opaque exact process identity>","port":49152}
```

PID plus process-start identity names one exact OS process occurrence. The port names that
occurrence's loopback control endpoint. The row identifies the latest admitted occurrence but does
not establish that it remains alive. A stale row has no serving or version authority; it remains
only as the exact predecessor value used for atomic replacement.

Revision belongs to observable live health, not durable coordination. A revision orders two live
ACN targets: an older client may use an equal or greater live revision, while a newer client must
replace a lower live revision. No revision continues to constrain launch after its process tree is
absent.

## Required store operations

Every version implements exactly these semantic operations:

```text
currentOwner
  -> None | Owner

replaceOwner(expectedOwner: None | Owner, candidateOwner)
  -> Replaced
   | OwnerChanged(None | currentOwner)
```

`replaceOwner` is exactly one short atomic transaction:

1. execute `BEGIN IMMEDIATE`;
2. read the complete current owner;
3. if it differs from `expectedOwner`, roll back and return `OwnerChanged`;
4. insert or replace the singleton row with `candidateOwner`;
5. commit and return `Replaced`.

The transaction contains no HTTP, process inspection, sleep, Effect suspension, artifact work,
application callback, or process launch. Atomic commit exposes either the complete predecessor or
the complete candidate. Two candidates that observed the same predecessor cannot both commit.

## Required owner endpoint

The exact live process recorded as owner serves:

```text
GET  http://127.0.0.1:<owner.port>/health
POST http://127.0.0.1:<owner.port>/shutdown
```

Health contains at least the exact owner PID and the process's revision. HTTP `200` means ready for
new clients; HTTP `503` means live but not ready. Additional fields are optional diagnostics and
cannot establish liveness or extend a deadline.

Shutdown atomically closes work admission, enters the monotonic stopping lifecycle, and returns
without waiting for drain, finalizers, child shutdown, or process exit.

## Live convergence

A manager interprets coordination only after combining the owner row with exact process inspection
and health:

- If the exact owner tree is absent, the row has no authority and may be replaced.
- If the root is absent but descendants remain, the exact predecessor tree is retired before any
  successor may acquire ownership.
- If health is temporarily unavailable, the manager waits only for the bounded health grace before
  retiring the exact owner tree.
- If live health reports a revision equal to or greater than the client's target, the client waits
  for that occurrence to become ready and adopts it. An older client never replaces a newer owner.
- If live health reports a lower revision, the client first prepares its own launch material, then
  retires that exact owner tree and competes to install its candidate.
- Ready adoption rereads the same owner row and exact process identity after observing health.

Candidate launch and admission remain separate. Before admission, a candidate binds only its
health/shutdown endpoint, starts no application or ICN service, and remains parent-bound and
scope-owned by its manager. It rereads the expected owner, proves the predecessor tree absent, and
calls `replaceOwner`. `OwnerChanged` rejects admission and the candidate exits. `Replaced` transfers
ownership to the row and permits application startup.

Concurrent candidates need no persisted version election. Atomic owner replacement chooses one
occurrence. If a lower revision wins, a live newer manager subsequently replaces it; if a higher
revision wins, older managers adopt it. Process death removes revision authority automatically.

## Guarantees

- The exact owner row is the only durable coordination fact.
- No stale row or historical revision can block launch.
- No candidate starts application or ICN work before atomic owner admission.
- Two candidates observing the same predecessor cannot both commit.
- A predecessor row is replaced only after exact predecessor-tree absence proof.
- Every raw candidate is scope-owned until exact owner publication.
- A lower live revision is replaced only after successor launch material is prepared.
- An equal or greater live revision is never downgraded by an older client.
- No stale manager action targets a changed owner or reused PID.
- Observation uncertainty authorizes neither endpoint adoption nor overlapping service trees.
- SQLite contention and every convergence state are bounded.
