---
applies_to:
  - packages/acn/src/activity-tracker.ts
  - packages/acn/src/resource-use-gate.ts
  - packages/acn/src/acn-shutdown.ts
  - packages/acn/src/acn-subscriptions.ts
  - packages/acn/src/acn-subscription-protocol.ts
  - packages/acn/src/agent-runtime.ts
  - packages/acn/src/session-commands.ts
  - packages/acn/src/session-lifecycle.ts
  - packages/acn/src/session-drafts.ts
  - packages/acn/src/active-session-statuses.ts
  - packages/acn/src/display-view-streams.ts
  - packages/acn/src/handlers.ts
  - packages/acn/src/ops.ts
  - packages/acn/src/server.ts
  - packages/protocol/src/rpcs/**
  - packages/protocol/src/schemas/subscription.ts
  - packages/protocol/src/schemas/display.ts
  - packages/protocol/src/schemas/shell.ts
  - packages/sdk/src/jit-rpc/**
  - packages/sdk/src/acn-jit/**
  - packages/client-common/src/**
  - packages/agent/src/session-work-status.ts
  - packages/agent/src/coding-agent.ts
  - packages/agent/src/compaction/worker.ts
  - packages/agent/src/events.ts
  - packages/agent/src/display/**
  - packages/agent/src/display-view/**
  - packages/agent/src/model/model-resolver.ts
  - packages/agent/src/index.ts
  - packages/agent/src/execution/execution-manager.ts
  - packages/agent/src/execution/types.ts
  - packages/agent/src/process/detached-process-registry-live.ts
  - packages/agent/src/process/detached-process-registry.ts
  - packages/agent/src/projections/compaction.ts
  - packages/agent/tests/session-work-status.test.ts
  - cli/src/**
  - desktop/src/**
  - web/src/**
---

# ACN and session lifetime

ACN and session runtimes stay alive while they are doing work, then shut down independently after
an idle period.

## ACN lifetime

ACN shuts down after 30 minutes without work. The first idle period begins only after ACN, its HTTP
server, registration, and private ICN are ready.

Finite RPCs keep ACN alive for the full request. Work that continues after a request—such as an
agent turn or model operation—keeps its own claim until it finishes.

Observation does not keep ACN alive. This includes health checks, subscriptions, status and file
watches, mirrored-state refresh, display streams, ICN observation, telemetry, and introspection.

All stop causes use the same shutdown path: stop accepting work, close the application, terminate
and reap ICN, then release machine ownership. Closing the application closes resident session scopes
directly and cancels their work; ACN shutdown never waits for a session idle-retirement gate whose
leases are owned by those same scopes.

## Session runtime lifetime

A session runtime unloads after two minutes without session work. Commands, agent execution,
display materialization, shape changes, resync, and preload count as work. Merely watching a session
does not.

Agent work has one authoritative status covering turns, queued triggers, workers, compaction, and
owned detached processes. Both UI status and runtime lifetime use that status.

Session startup is single-flight. Unloading closes the current runtime before publishing it as
absent; later work creates a new runtime. A draft stores session intent, not a runtime. Deletion
blocks new work, waits for current work to finish, closes the runtime, then deletes durable state.

Preloaded and resident sessions consume one ACN-owned, revisioned model configuration. A slot
mutation publishes its new configuration before it succeeds. Before a session accepts an external
event, it synchronizes to the latest published revision; delayed subscription delivery or an older
queued revision cannot replace a newer configuration. Selecting a model therefore makes an
already-preloaded draft immediately usable without restarting the client or daemon.

## User bash command history

A user bash command is session work. Completion records one identified session event containing the
command, working directory, exit code, and bounded stdout and stderr. That event is the sole source
of truth for agent context and display history; clients must not maintain a parallel command-result
history.

Display replay, pagination, and session resume reproduce each recorded command exactly once. A
command issued during an active agent turn remains in the pending user-activity suffix so later
output from that turn is displayed before the command. Interrupting the turn restores queued text
to the composer but retains already-executed bash commands in history.

## Display streams during session unload

The display subscription belongs to ACN, while its live display attachment belongs to the current
session runtime. An open display subscription therefore does not keep the session loaded.

When the session unloads, ACN detaches the live display and tells the subscription it is suspended.
The client keeps the last display state. A later materialization, shape change, or resync reloads the
session, reattaches the display, and sends a full snapshot.

Detachment invalidates the attachment generation before signaling its forwarding fiber to stop.
Retirement does not wait for that fiber's downstream finalizers: the generation token prevents any
late event from being published, while cleanup completes asynchronously.

Active local-model request progress is transient session-runtime state, not an app event or chat
message. It is keyed by fork and included in every live display snapshot, so a reconnect sees the
current request while it remains active. It disappears when generation begins or the request
stream ends and is never written to session history. Clients may delay rendering briefly to avoid
flashing short prefills, but they do not infer token progress or preserve a second copy.

Model-request progress enters the event engine as timestamped ambient observations through one
progress sink shared by ACN preparation and provider execution. A display-owned projection reduces
those observations into the current per-fork request activity and response timing. Display views
read that projection through the normal projection-consumer path, so activity changes invalidate
and rematerialize snapshots without an app event or a separately merged runtime stream. Projection
evaluation never reads a clock; observation time is fixed at ambient ingress.

The CLI reserves one fixed-height activity rail below the chat timeline. Root-model loading,
conversation prefill, and model response activity all occupy that same row, so transitions do not
move the chat or composer. Model loading is shown immediately. Request preparation and prefill are
shown only after a short anti-flicker delay. Once generation begins, the row becomes the existing
composable Working display, including thinking, tools, advisor activity, and worker counts.

The CLI composer keeps a one-line minimum editor inside a full-width input surface with one stable
row of visual padding above and below it and a tapered mode rail painted within the surface's first
column; additional visual input lines grow only from actual draft content. Pending attachments render inside that surface rather than competing with persistent
status. The terminal-background footer below it derives model selection, reasoning effort, local
residency, working directory, and context usage directly from authoritative mirrors and display
state. Model and effort form one visual identity, separated from resident memory by a muted middle
dot. Local residency uses distinct loaded, transitional, and unloaded glyphs; detailed load
percentage and failures remain in the activity rail.

While the root slot is loading, the rail may present its authoritative percentage as well as
persistent model chrome. Both presentations derive from the one mirrored slot lifecycle state;
readiness hands the rail from model loading to request prefill without a client-owned loading state
or timer. A typed low-memory load failure or resident-runtime loss uses that same slot state to
replace progress with the durable low-memory stopped message until the user retries or changes the
selection.

Local request preparation publishes preparing activity before acquiring the selected model. Its
request identifier remains absent until ICN accepts the native request, and that handoff preserves
the activity start time. If preparation or provider start fails, orchestration clears the activity;
after provider acceptance the provider owns all later progress and clearing. Consequently a long
model load satisfies the client's anti-flicker delay and the rail transitions directly into
conversation loading when the slot becomes ready.

The activity projection records the generation-start timestamp in the same atomic transition that
removes prefill progress. Display snapshots attach that timestamp to root work for the current
chain. The live Working timer therefore begins when the model starts responding, rather than
including model admission or prefill time, and the rail can transition from prefill to Working
without disappearing between snapshots. Providers that do not expose granular request progress
continue to use the actor's work-start timestamp as a fallback.

Closing the final subscription removes the display registration. There is no separate close RPC.

## Subscription protocol

ACN subscriptions wrap domain values in a small transport protocol:

| Frame | Meaning |
| --- | --- |
| `payload` | Domain value |
| `keepalive` | Connection is alive |
| `suspended` | Session runtime unloaded; subscription stays open |
| `terminated` | ACN is shutting down |

Framing is handled below RPC handlers and client consumers. Invalid frames, or a stream ending
without `terminated`, are protocol errors rather than evidence that ACN died.

## Concurrency guarantees

Work is represented by scoped claims tied to a specific ACN or session-runtime generation. The last
claim starts that generation's idle timer. Starting work and committing shutdown are serialized, so
work cannot be admitted into a generation that is being destroyed. Stale or duplicate cleanup from
an older generation cannot affect its replacement.

Resolving a session under the runtime admission lock never waits for that session's retirement.
This prevents one wedged generation from blocking unrelated sessions. A command that arrives behind
an abnormally long retirement fails after a bounded wait, before its session event is accepted, so
the client can restore the user's unsent input instead of waiting forever. Retirement that remains
stalled beyond the process liveness deadline requests controlled ACN replacement; the old gate is
never rolled back into a partially closed generation.
