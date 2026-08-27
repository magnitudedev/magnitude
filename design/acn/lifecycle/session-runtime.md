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
Absent -> Starting -> Loaded -> Retiring -> Absent
```

## Admission and lifetime

Startup is single-flight per session. There are two semantic entities: the durable `Session` and
its currently loaded, disposable `SessionRuntime`. A runtime's generation is only an exact-instance
fence; cleanup from an old generation cannot affect its replacement.

Callers acquire a `SessionRuntime` in an Effect scope. Commands, preload, and display
materialization (opening or reopening a display subscription) are bounded scoped uses. Holding an
open display subscription does not retain the runtime. Scope finalization releases the use for
every Effect exit, without callback wrappers or caller-selected work classifications.

Agent work has one authoritative status covering accepted messages awaiting resolution, turns,
queued triggers, workers, compaction, and owned detached processes. Each runtime generation owns
no additional work claim. The runtime stays loaded while its scoped-use count is nonzero or this
aggregate status is `Working`. It becomes idle only when the scoped-use count is zero and the status
is `Quiescent`, then unloads after two minutes if neither fact changes.

ACN subscribes to the current-first aggregate work status before publishing a newly loaded runtime.
Status changes update the runtime's private lifetime state directly. Before final release and
retirement, ACN rereads the authoritative status so a delayed notification cannot unload working
state. There are no per-message, per-worker, display, continuing-work, demand, or process-global
claims.

Resolving a session under the runtime admission lock never waits for that session's retirement, so
one wedged runtime cannot block unrelated sessions. Work arriving behind abnormal retirement fails
after a bounded wait and before accepting a session event, allowing the client to restore unsent
input. A partially closed runtime is never reopened. Retirement failure is contained to that exact
session generation: it cannot stop ACN, ICN, model acquisition, inference, another session, or
another client.

## Drafts, creation, and deletion

A draft stores session intent, not a runtime. Preload and claim phases are outcome-total:
cancellation removes a preloading record or restores a claim. Claiming linearizes session creation;
initial message or goal publication and draft promotion or rollback then complete independently of
client interruption.

Deletion closes new work, waits for accepted work, retires the runtime, and only then removes durable
state. Claiming deletion atomically transfers it to a manager-scope fiber; the requesting caller
then waits interruptibly and may detach without canceling accepted cleanup. Retirement itself
remains interruptible so its liveness race can cancel the losing watchdog, and only the brief claim
and ownership transfer are masked. ACN shutdown instead closes every resident runtime scope
directly.

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
- Every accepted work item belongs to exactly one live `SessionRuntime`.
- A `SessionRuntime` remains loaded exactly while it has a scoped user or aggregate work is
  `Working`, followed by its configured idle interval.
- Admission and retirement cannot cross, and old cleanup cannot affect a replacement.
- A stuck or failed retirement quarantines only its exact session generation and cannot determine
  process lifetime.
- Draft cancellation cannot strand preloading or claiming.
- Deletion accepts no work after its commit point.
- Display recovery cannot publish late events from a retired generation.
