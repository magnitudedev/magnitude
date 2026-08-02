---
applies_to:
  - packages/storage/src/io/storage.ts
  - packages/storage/src/sessions/**
  - packages/acn/src/agent-persistence.ts
  - packages/agent/src/persistence/**
  - packages/agent/src/runtime/projection-snapshot-hydration.ts
  - packages/agent/src/workers/lifecycle-coordinator.ts
  - packages/event-core/src/core/event-sink.ts
---

# Session event-log durability

The session event log is the authoritative, irreplaceable record from which agent state is
rehydrated. It is an append-only UTF-8 JSONL file. Projection snapshots are disposable caches and
never replace the event log as recovery authority.

## Record boundaries

Every persisted event is compact JSON followed by one LF byte. LF is the durable record boundary
used by automatic recovery. Embedded newlines are JSON-escaped and therefore do not create physical
record boundaries.

A syntactically valid JSON value after the final LF is a complete event missing only its delimiter.
A syntactically invalid fragment after the final LF is an incomplete append. Reads preserve the
former by adding LF and discard the latter by truncating to the preceding LF.

Only an unterminated tail is automatically repairable. Every LF-terminated record is committed
history. Invalid UTF-8 or invalid JSON in a committed record—including an empty or whitespace-only
record—is a visible persistence failure and is never skipped or deleted automatically.

The generic JSONL layer validates UTF-8 and JSON syntax. Its type parameter describes the caller's
expected value but is not runtime validation. Magnitude does not currently define a runtime schema
for the complete `AppEvent` union, so event-variant shape validation is outside this recovery
boundary. A syntactically valid committed JSON value is never deleted as tail damage merely because
a downstream domain consumer cannot use its shape.

Repair decisions use byte offsets in the UTF-8 file. Event content is not included in recovery
diagnostics.

## Unified file ownership

One path-scoped, Effect-native JSONL abstraction owns all session event reads, tail repair, appends,
record counting, and process-local synchronization. Callers do not independently compose raw file
reads, path locks, and appends.

Reads and appends for the same normalized path serialize on one semaphore. A successful read leaves
the physical file at a valid LF boundary, not merely a logical view that ignores damaged bytes.
Append validates or repairs the tail before adding records.

The abstraction may retain the known byte length and committed record count in memory so cursor
calculation remains proportional to the appended batch. Missing or mismatched metadata is rebuilt
from the authoritative file. The metadata is never persisted as a required sidecar and never
changes recovery semantics.

Process-local synchronization does not replace ACN runtime ownership. Only the active ACN runtime
may mutate session events. A future cross-process writer or repairing reader requires an explicit
ownership protocol before it is introduced.

## Interruption and commit

JSON encoding completes before filesystem mutation. Lock acquisition and encoding remain
interruptible. Once tail repair or append mutation begins, the bounded filesystem operation is
uninterruptible so Effect scope closure cannot abort Node `writeFile` after an arbitrary prefix has
been written.

Process death, host shutdown, and storage failure can still leave an unterminated tail. The next
read repairs that tail before returning, and the next append repairs it before writing. A damaged
tail is therefore never left in place for a later append to convert into committed interior
corruption.

Pending events remain in the event sink until the append commits. A scoped claim gives one flusher
exclusive ownership of an exact pending prefix; closing the scope without acknowledgement leaves
that prefix pending. Append and claim acknowledgement form one uninterruptible lifecycle critical
section. Failure or interruption before commit leaves the events pending in their original order.
After commit, only the claimed prefix can be acknowledged, and it can be acknowledged once. Events
published while the claim is open remain pending for the next claim.
Projection snapshot capture occurs afterward and may be interrupted because the event log remains
authoritative.

## Data-class separation

Session logs and trace events are diagnostic JSONL, not authoritative session history. Their
high-frequency synchronous appenders do not acquire event-log recovery machinery or rewrite their
files. Shared encoding utilities do not imply shared durability policy.

## Acceptance criteria

- Reading an invalid unterminated tail truncates it to the preceding LF and returns the valid
  committed prefix.
- Reading a complete final JSON value missing LF preserves it and adds LF.
- A successful read is physically safe for immediate append.
- Appending never rewrites or swaps the committed event-log prefix.
- Session retirement waits for an in-progress append filesystem mutation instead of aborting it.
- An uncommitted pending batch is not lost when its flush is interrupted.
- Cursor index and timestamp refer to the final committed event after recovery and append.
- Committed corruption remains visible and leaves the file unchanged.
- Steady-state append and cursor calculation do not parse the complete log after its record count is
  known.
