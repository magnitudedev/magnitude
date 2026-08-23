---
applies_to:
  - packages/acn/src/agent-runtime.ts
  - packages/acn/src/session-*.ts
  - packages/acn/src/active-session-statuses.ts
  - packages/acn/src/display-view-streams.ts
  - packages/acn/src/agent-persistence.ts
  - packages/acn-protocol/src/boundary/shell.ts
  - packages/acn-protocol/src/schemas/shell.ts
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

A resident runtime unloads after two minutes without session use. Commands, preload, and display
materialization (opening or reopening a display subscription) are bounded uses. Holding an open
display subscription does not retain the runtime.

Agent work has one authoritative status covering accepted messages awaiting resolution, turns,
queued triggers, workers, compaction, and owned detached processes. Each runtime generation owns
at most one continuing-work claim for that aggregate status. Message and goal admission establish
the claim before committing work; a later `Working` transition confirms it, and the corresponding
`Quiescent` transition releases it and starts a fresh idle interval. Rehydration of an already
working session establishes the same claim before publishing the resident generation. Runtime
ownership and UI consume the aggregate status instead of reconstructing work independently; there
are no per-message, per-worker, display, or process-global work claims.

Resolving a session under the runtime admission lock never waits for that session's retirement, so
one wedged generation cannot block unrelated sessions. Work arriving behind abnormal retirement
fails after a bounded wait and before accepting a session event, allowing the client to restore
unsent input. Persistent retirement failure requests controlled ACN replacement; the old gate is
never reopened into a partially closed generation.

## Drafts, creation, and deletion

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

The display subscription (`StreamDisplayView(sessionId, shape)`) belongs to ACN; one view exists
per session and shape, and its live attachment belongs to one runtime generation. Opening a
subscription materializes its view — loading the runtime if needed, setting the view's shape, and
emitting a complete snapshot first — and every later subscriber of the same view rereads a complete
snapshot. Unload invalidates the attachment generation before stopping its forwarding fiber; it
does not wait on downstream finalizers, and late output from the old generation is rejected while
cleanup finishes asynchronously. The subscription stays open through unload.

The client retains its last display state. Reopening the subscription (a shape change, resync, or
retry) or work that makes the session busy again reattaches the display and emits a complete
snapshot.

## Guarantees

- Durable session state remains authoritative across unload, restart, and ACN replacement.
- Observation alone cannot retain a runtime.
- Every accepted work item belongs to exactly one live runtime generation.
- Every resident generation has at most one continuing-work claim, owned before work commit and
  released only after aggregate quiescence.
- Admission and retirement cannot cross, and old cleanup cannot affect a replacement.
- Draft cancellation cannot strand preloading or claiming.
- Deletion accepts no work after its commit point.
- Display recovery cannot publish late events from a retired generation.
