---
applies_to:
  - packages/acn/src/*.ts
  - packages/client-common/src/headless/session-runner.ts
  - packages/sdk/src/headless-rpcs.ts
  - packages/acn-protocol/src/*/shell.ts
  - packages/agent/src/events.ts
  - packages/agent/src/session-work-status.ts
  - packages/agent/src/process/detached-process-registry*.ts
---

# Session runtime lifecycle

A session runtime is an ACN-owned disposable execution environment over durable session state. It
is neither session identity nor history and can unload and reconstruct without changing either.

```text
Absent -> Starting -> Resident -> Retiring -> Absent
```

## Admission and residency

Startup is single-flight per session. Work enters through a generation-scoped gate: retirement
closes admission before publication, and cleanup from an old generation cannot affect its
replacement.

A resident runtime unloads after two minutes without session work. Commands, agent execution,
display materialization, shape changes, resynchronization, and preload count as work. Merely
watching a session does not. The final claim starts that generation's idle timer.

Agent work has one authoritative status covering accepted user messages awaiting resolution,
turns, queued triggers, workers, compaction, and owned detached processes. Runtime retention and UI
consume that status instead of reconstructing work independently. The transition from accepted
input to a queued or active turn therefore cannot expose a false quiescent interval.

Resolving a session under the runtime admission lock never waits for that session's retirement, so
one wedged generation cannot block unrelated sessions. Work arriving behind abnormal retirement
fails after a bounded wait and before accepting a session event, allowing the client to restore
unsent input. Persistent retirement failure requests controlled ACN replacement; the old gate is
never reopened into a partially closed generation.

## Drafts, creation, and deletion

A non-interactive client chooses a branded session ID before dispatching `CreateSession` and flushes
that receipt before the RPC can leave the process. Ambiguous transport termination therefore never
loses the durable recovery key. Its client-specific SDK RPC group decodes every daemon result with
strict excess-property rejection before generic RPC normalization. A `Worked` root is terminal only
when the authoritative root timeline contains a complete ordered success or failure record; missing,
empty, or internally inconsistent `messages.order` remains nonterminal and fails closed if the stream
ends.

A draft stores session intent, not a runtime. Preload and claim phases are outcome-total:
cancellation removes a preloading record or restores a claim. Claiming linearizes session creation;
initial message or goal publication and draft promotion or rollback then complete independently of
client interruption.

Deletion closes new work, waits for accepted work, retires the runtime, and only then removes durable
state. ACN shutdown instead closes every resident runtime scope directly.

## Runtime configuration

Preloaded and resident runtimes consume one ACN-owned model configuration. Slot mutation publishes
the new configuration before success. Subscription emits the current value first and then semantic
changes. Before accepting an external event, the runtime rereads the current value under its
synchronization boundary and updates ambient state only when meaning changed. Delayed observation
therefore cannot overwrite newer configuration, and an already-preloaded draft becomes usable after
model selection without client or daemon restart.

## Durable shell commands

A completed user shell command commits one identified session event containing command, working
directory, exit code, and bounded stdout/stderr. That event alone supplies agent context and display
history; no client keeps a parallel result history. Replay, pagination, resume, and runtime rebuild
reproduce it exactly once.

A command accepted during an active turn remains in the pending user-activity suffix, after later
output from that turn. Interrupt restores queued text to the composer but never removes an executed
command.

## Display attachment

The display subscription belongs to ACN; its live attachment belongs to one runtime generation.
Unload invalidates the attachment generation before stopping its forwarding fiber, then emits
session suspension through the [ACN subscription protocol](../subscriptions.md). It does not wait
on downstream finalizers. Late output from the old generation
is rejected while cleanup finishes asynchronously.

The client retains its last display state. Later materialization, shape change, or resync reloads the
runtime and serializes one atomic shape-plus-snapshot commit per display registration. The accepted
full state crosses a subscriber admission fence before any live update; stale events from the prior
shape or runtime generation are rejected.

## Guarantees

- Durable session state remains authoritative across unload, restart, and ACN replacement.
- Observation alone cannot retain a runtime.
- Every accepted work item belongs to exactly one live runtime generation.
- Admission and retirement cannot cross, and old cleanup cannot affect a replacement.
- Draft cancellation cannot strand preloading or claiming.
- Deletion accepts no work after its commit point.
- Display recovery cannot publish late events from a retired generation or pre-commit shape.
