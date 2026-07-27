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

Operation cancellation follows semantic ownership rather than the initiating transport. Shared
work admitted by a domain service is owned in that service's scope; losing one request cancels only
that request's wait. Finite mutations become outcome-total at their linearization point. The
complete classification is defined by [operation ownership](./operation-ownership.md).

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
Draft preload and claim phases are outcome-total: cancellation removes or restores the phase.
Claiming linearizes creation; initial message or goal publication and draft promotion or rollback
then complete before client interruption can take effect.

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
those observations into the current per-fork request activity. Display views
read that projection through the normal projection-consumer path, so activity changes invalidate
and rematerialize snapshots without an app event or a separately merged runtime stream. Projection
evaluation never reads a clock; observation time is fixed at ambient ingress.

The CLI timeline follows appended tail content while the user remains attached to the bottom. New
messages do not reposition a short page: they occupy the next available rows, and the viewport moves
only by the amount that real content overflows it. Manual scrolling detaches this behavior. The
terminal scroll surface provides the tail-following primitive, while the shared scroll controller
continues to own history loading, prepend anchoring, and overlay restoration.

While a root work chain is active, the live activity rail occupies a reserved bottom slot outside
the scrollback and directly above the composer. It therefore remains at the bottom on short pages as
well as overflowing ones; assistant messages and tool output grow and scroll independently above it.
One blank terminal row separates history from the rail above it. No full blank row follows the live
activity label. Instead, the terminal-background label begins with a cyan `┏━` branch whose
downstroke meets the composer's left border on the immediately following row. The composer's own
top padding provides separation from the editor without an additional connector row. This makes
the activity a header of the input rather than part of conversation history. Root-model loading,
conversation prefill, and model response activity share that same stable label. Model loading is shown
immediately. Request preparation and prefill use a short anti-flicker delay. Once generation begins,
the row becomes the composable Working display, including thinking, tools, advisor activity, and
worker counts. Manual history detachment never moves or hides the live rail and never causes the
rail to force the history back to its tail.

When the root work chain becomes stable, the transient rail is replaced at the same tail position by
a durable work summary immediately after the chain's final assistant, tool, or worker output and
before any queued follow-up user activity. A chain spanning multiple internal turns or workers
produces exactly one summary. Its duration accumulates the union of productive root generation,
tool execution, and worker time, while excluding intervals spent only on model admission, model
loading, conversation prefill, or retry waits. The summary is projected from session events,
survives replay and pagination, and scrolls with conversation history. Starting a later chain
creates a new transient rail at the new live tail; clients never preserve completed rail state
independently.

When native generation performance is available, the durable summary also records the root model's
display name and decode throughput. A chain with one root request preserves the provider-reported
decode rate unchanged. A chain with multiple root requests weights throughput by native
generated-token counts and decode durations; it never averages request rates or includes model
loading, prefill, tools, workers, or wait time in tok/s. Time to first token remains request-level
diagnostic data and is not included in durable chat summaries.
Worker generations do not enter the root-model performance aggregate because they may use a
different model. Providers that report no native performance produce the ordinary duration-only
summary, without client-side timing, tokenization, or provider-identity branching.

The startup identity is a non-persisted prefix at the true beginning of the timeline. It remains
the first visual content, uses a proportionally reduced complete Magnitude mark beside its identity
and directory details when width permits, stacks at narrow widths, and scrolls away naturally with
conversation history. It is not a chat message and is absent when the loaded window does not reach
the beginning of the session.

After first-run onboarding selects a local model whose packages are still installing, the startup
identity remains above a bordered download card centered in the otherwise empty history area.
Recent conversations are suppressed until installation and preparation finish. The card derives
model identity, quantization, stage, byte progress, transfer rate, and remaining-time presentation
from authoritative local-model mirrors; it is not a session event or activity rail. Its confirmation
state is presentation-only, while confirmed cancellation is one server mutation that returns to the
onboarding chooser. The composer remains visible but cannot submit while the selected model is not
usable.

The CLI composer keeps a one-line minimum editor inside a full-width input surface with one full
row of visual padding above and below it. A solid mode border sits immediately outside the colored
surface and spans its full height; additional visual input lines grow only from actual draft
content. Pending attachments render inside that surface rather than competing with persistent
status. The terminal-background footer below it derives model selection, reasoning effort, local
residency, working directory, context usage, and resident memory directly from authoritative
mirrors and display state. Model, effort, context usage, and resident memory form the left group
with stable spacing; the working directory is the only right-aligned item. Local residency uses distinct loaded,
transitional, and unloaded glyphs; detailed load percentage and failures remain in the activity
rail.

Reasoning effort is a violet interactive label rather than muted metadata. Activating it opens a
horizontal selector immediately to its right in the footer; the selector replaces context
temporarily while the working directory remains the sole right-aligned item. Available
choices come from the selected model's catalog control. The selector maintains only an ephemeral
preview: repeated Ctrl-T, left/right arrows, and other directional keys cycle with wraparound;
Enter or a clicked choice
commits through the existing model-configuration mutation, and Escape closes without changing the
saved effort. It has no timeout and never renders inside the colored composer surface.

When the selected local-model slot is ready and its server-published resident allocation is
available, the composer footer shows one compact memory value exactly three spaces after context,
for example `24 GB mem`. The value sums model, context, compute, and auxiliary allocations across
participating memory domains. It uses the same muted presentation as context and is plain,
non-interactive text. Memory disappears completely for cloud, loading, unloading, unloaded,
failed, or unavailable states, and while the reasoning selector temporarily replaces context.
There is no separate local-inference badge or client-owned transition state.

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

Root work is a durable phase machine. A root turn begins in a waiting-for-model phase. The execution
adapter records one generation-start event for the turn before its first semantic thinking,
assistant-message, or tool-input event; this moves the root into working and starts the clock.
A chain-continuing outcome closes that productive interval and returns to waiting-for-model before
the next turn. An active worker keeps the productive clock running across that root-model wait; if
the final worker settles before generation begins, the clock pauses until generation. A terminal
root outcome either completes the chain or moves to waiting-for-workers, whose elapsed time remains
productive until the final root worker settles. Interrupts close the current interval without
manufacturing time for a wait-only turn.

The root projection owns accumulated productive milliseconds and the current running interval.
Display snapshots expose that authoritative state, and clients only add the current clock delta
while that interval is open. Transient request progress controls loading and prefill copy but never
owns, reconstructs, or adjusts the work timer. This keeps live and replayed completed timing
identical even when a chain crosses several model requests.

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
