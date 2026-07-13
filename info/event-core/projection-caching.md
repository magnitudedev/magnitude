# Projection Caching

Projection snapshots are a startup cache. The event log is still the source of
truth, and startup can always fall back to replaying events when a snapshot is
missing, stale, invalid, or unwritable.

## Core Invariant

A persisted projection snapshot must describe projection state after all events
up to and including its `eventCursor`, and before any later event.

The cursor is not guessed from memory. It comes from the storage append that
writes pending events to the event log:

- `index` is the zero-based index of the last appended event in the full log.
- `timestamp` is the timestamp of that last appended event.

## Trigger Policy

`LifecycleCoordinator` owns durability timing. Projection snapshotting does not
have a separate scheduler or policy.

The lifecycle coordinator flushes persistence in the same cases as the existing
event persistence system:

- after `session_initialized`, it starts the periodic flush timer,
- on each periodic timer tick, currently every 1.5 seconds,
- shortly after `TurnProjection` emits `turnTerminated`,
- once more when the engine scope shuts down.

The only added behavior is that a successful event flush also captures and saves
a projection snapshot at the cursor returned by that event flush.

## Flush Semantics

Each lifecycle flush runs at a stable event-bus boundary. That boundary is a
synchronization mechanism, not a new trigger policy: it makes the flush run after
previously published events have updated projections and entered the pending
event sink, and before later events update projections.

Inside that boundary, the flush does this:

1. Drain pending non-ephemeral events from the event sink.
2. If there are no pending events, stop.
3. Read session metadata for the session id.
4. Append the pending events to the event log.
5. Use the returned append cursor as the snapshot cursor.
6. Capture every registered projection's encoded snapshot at that cursor.
7. Atomically write the projection snapshot envelope.

This keeps the event cursor and projection state aligned. Without the boundary,
a snapshot could include an event that was processed by projections but not yet
included in the cursor written to disk.

## Addressed Projection State

Addressed projection snapshots contain ordinary addressed indexes, not addressed
entry bodies. Before snapshot capture encodes ordinary projection state, dirty
addressed entries are flushed so those indexes resolve through the entry store.
This is one of exactly three flush points — the others are the release of an
entry's last pin and the write-through of entries a commit leaves unpinned.
Nothing flushes on the event-bus cycle; durability between flush points is the
event log's job.

This matters because addressed entry files are physical storage for large
projection-owned values. A snapshot must not restore ordinary indexes from one
event-log point against addressed entry files from an incompatible later point.
If restore cannot prove the snapshot cursor is safe, startup falls back to event
replay.

Old physical addressed entries may remain after structural sequence rewrites so
previously captured windows stay readable. Their lifecycle belongs to explicit
compaction or garbage collection, not to snapshot capture.

See [Addressed Projection State](./addressed-state.md) for the sequence identity,
physical-entry reuse, and residency invariants.

## Failure Semantics

If metadata loading or event append fails before events become durable, the
drained events are put back at the front of the pending sink. A later lifecycle
flush can retry them without losing ordering.

If event append succeeds but snapshot capture or snapshot write fails, the
events are not requeued. Requeueing already-appended events would duplicate the
event log. In that case the event log remains correct, and startup can recover
by replaying from the previous valid snapshot or from the beginning.

Snapshot writes are atomic at the file level. Event-log append and snapshot write
are not treated as one cross-file transaction; correctness comes from the event
log being authoritative and snapshot restore verifying its cursor.

## Restore Path

On startup, the agent first tries to load a projection snapshot. If loading or
validation fails, it replays the full event log.

If a snapshot loads, the agent checks that the event log still contains the
cursor event at the recorded index and timestamp. If the cursor does not match,
the snapshot is stale or foreign and the agent replays the full log.

If the cursor matches, projection restore validates the snapshot envelope:

- engine name must match,
- schema version must match,
- every snapshot key must correspond to a registered projection,
- every registered projection must have a snapshot entry,
- each projection must be able to decode and prepare its own snapshot state.

Restore is all-or-nothing. Each projection prepares its restore before any
projection commits. If any projection fails validation, no partial projection
restore is committed and the agent falls back to event replay.

After a successful restore, the agent replays only the event suffix after the
snapshot cursor.

## Unknown Projection Keys

The snapshot envelope decodes `projections` as a string-keyed record of unknown
values. It does not decode `projections` as a struct of known projection names.

That matters because struct decoding strips excess keys. Keeping keys intact
lets restore explicitly reject stale or foreign snapshot entries that contain
unknown projection names.

## Scope

The lifecycle flush is deterministic inside one running agent process because
event publishing and lifecycle flushes share the event bus boundary.

This system does not provide cross-process file locking. Two separate agent
processes writing to the same session directory at the same time still require a
storage-level session lock.
