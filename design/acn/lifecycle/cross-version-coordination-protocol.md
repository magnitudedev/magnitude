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
Implementations may expose additional diagnostics, but another version must not require them.

Nothing outside this document is a cross-version coordination prerequisite.

## Filesystem and database surface

For Magnitude data root `D`, the complete shared filesystem surface is:

```text
D/acn/coordination.sqlite
```

The database contains exactly these required semantic tables:

```sql
CREATE TABLE revisions (
  revision INTEGER PRIMARY KEY
    CHECK (revision > 0 AND revision <= 9007199254740991)
);

CREATE TABLE owner (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  pid INTEGER NOT NULL CHECK (pid > 0 AND pid <= 9007199254740991),
  process_start_identity TEXT NOT NULL
    CHECK (length(process_start_identity) > 0),
  port INTEGER NOT NULL CHECK (port BETWEEN 1 AND 65535)
);
```

There are no revision files, development-specific records, `owner.json`, `owner-lock.sqlite`,
fixed port, persisted instance history, persisted workflow or lifecycle, owner generation, owner
lease, or compatibility representation.

Every connection uses SQLite rollback-journal mode and `busy_timeout=0`. `SQLITE_BUSY` is the only
contention result and callers retry it only within a fixed operation deadline. No connection,
transaction, SQLite implementation file, timestamp, or row identifier represents ACN ownership.

## Revision values and operations

A revision is one positive safe integer. Development computes its scalar before registration and
the protocol treats it exactly like a release revision.

`revisions` is append-only. A registered revision is never updated or deleted.

Development registration is permanent too. There are no active-development holds, and selection
never regresses merely because a development process exits or an older development revision is run
again. Registration is:

```sql
INSERT INTO revisions (revision) VALUES (?)
ON CONFLICT(revision) DO NOTHING;
```

Only an equal existing revision is ignored. The selected revision is:

```sql
SELECT MAX(revision) FROM revisions;
```

Zero rows means no selected revision. Invalid values, incompatible schema, or SQLite failure are
typed failures and never mean absence.

## Owner value

`owner` contains zero or one row. The only valid row has `id = 1`. Its complete semantic value is:

```json
{"pid":1234,"processStartIdentity":"<opaque exact process identity>","port":49152}
```

- `id` is only the SQLite singleton key. It is always literal `1`; it is not an occurrence,
  generation, or owner identity.
- `pid` is the positive safe-integer PID of the ACN root process.
- `processStartIdentity` is a non-empty opaque UTF-8 string identifying that exact OS process
  occurrence.
  Every version derives a byte-identical value for the same occurrence and compares it only by
  exact equality. Changing its derivation or encoding is a protocol change.
- `port` is the OS-assigned TCP port, from 1 through 65535, bound by that ACN on `127.0.0.1`.

The row is the complete latest admitted owner occurrence. A live process is not admitted unless the
row equals its complete owner value. An absent process may leave its row as exact predecessor
evidence until a successor replaces it. Revision is not an owner column; the admitted owner's
revision comes from its health response. No version may infer workflow or liveness from file
timestamps, SQLite implementation files, connection state, row IDs, or undocumented schema.

## Required store operations

Every version implements exactly these semantic operations:

```text
registerRevision(revision)

selectedRevision
  -> None | revision

currentOwner
  -> None | Owner

replaceOwner(expectedOwner: None | Owner, candidateOwner, candidateRevision)
  -> Replaced
   | OwnerChanged(None | currentOwner)
   | SelectionChanged(None | currentRevision)
```

`selectedRevision` and `currentOwner` are ordinary reads. Multiple owner rows, invalid values,
schema mismatch, or SQLite failure are typed failures. Their separate results are not one atomic
snapshot. A consumer re-observes when its required facts change and performs the action-specific
rereads required before mutation or ready adoption.

`replaceOwner` is exactly one short atomic transaction:

1. execute `BEGIN IMMEDIATE`;
2. read the selected revision and require exact equality with `candidateRevision`;
3. read the complete current owner and require exact equality with `expectedOwner`;
4. if selection differs, roll back and return `SelectionChanged`;
5. if owner differs, roll back and return `OwnerChanged`;
6. insert `id = 1`, or update its three owner columns on conflict, with the complete
   `candidateOwner`;
7. commit and return `Replaced`.

The transaction contains only bounded synchronous SQLite statements. It contains no HTTP, process
inspection, sleep, Effect suspension, artifact work, application callback, or process launch.
Atomic commit exposes either the complete old facts or complete new facts, never partial owner
data. No process holds the transaction for its owned lifetime.

Contention and store failure are bounded by the caller's fixed operation deadline. No SQLite result
other than `SQLITE_BUSY` is interpreted as contention.

## Required owner endpoint

The exact live process recorded as owner serves these routes on the dynamic `owner.port` value:

```text
GET  http://127.0.0.1:<owner.port>/health
POST http://127.0.0.1:<owner.port>/shutdown
```

`GET /health` returns JSON containing at least:

```json
{"pid":1234,"revision":42}
```

- `pid` equals the owner row PID.
- `revision` is the scalar revision admitted by that ACN.
- HTTP `200` means ready for new clients.
- HTTP `503` means the exact owner exists but is not ready for new clients.
- Additional fields are allowed. Another version may use them only as optional progress diagnostics;
  correctness and finite convergence may not require them.

`POST /shutdown` has no required body or identity header. The process receiving it atomically closes
work admission, enters its monotonic stopping lifecycle, and returns without waiting for work drain,
finalizers, child shutdown, or process exit.

Endpoint allocation and discovery are not fixed protocol constants: the candidate binds an
OS-assigned loopback port and publishes that port only through the owner row.

## Required process interactions

- A candidate prepares launch material before registering its revision.
- Before admission, a candidate derives its exact process identity and binds its control endpoint,
  but initializes no application or ICN service.
- A manager retires an unusable predecessor through bounded shutdown, TERM, KILL, and exact-tree
  absence proof. A candidate repeats the final exact-tree absence proof immediately before owner
  replacement.
- Every admitted ACN is the root of one dedicated OS process-tree termination unit, and every ICN it
  owns remains in that unit. Tree signaling and absence proof continue to address that unit if the
  ACN root exits while a descendant remains.
- A candidate calls `replaceOwner` with the complete owner it observed before absence proof and with
  its own revision and complete owner value.
- `OwnerChanged` or `SelectionChanged` rejects admission. That candidate exits without application
  or ICN startup.
- `Replaced` is admission. Only the recorded candidate may initialize application or ICN services
  or return readiness.
- The predecessor row is replaced only after its exact ACN/ICN tree is proven absent. Failure to
  prove absence never permits replacement.
- A serving ACN begins retirement after positively selecting a different required revision.
  Missing, unreadable, or indeterminate selection alone does not retire it.
- A client accepts an owner only after rereading the same row and proving: its exact process identity
  is live, health PID matches, health revision equals current selection, and health returns `200`.
- Before HTTP shutdown or every signal escalation, a manager rereads the owner row and exact process
  identity. A changed owner is never targeted by an action intended for its predecessor.
- A stale owner row has no serving authority, but remains the exact expected value for atomic
  successor replacement.
