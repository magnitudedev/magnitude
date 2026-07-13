# Display Architecture

Display is a negotiated, windowed runtime view over event-sourced session
state. The client owns display intent, the agent owns accepted display truth,
and ACN/SDK move commands and accepted snapshots between them.

The architecture is race-free because shape writes have one path, accepted
truth has one owner, and residency is handled by event-core instead of UI
caches or display events.

## Core Model

The event log is the durable source of truth. Displayable facts are derived
from events by projections.

`DisplayTimelineProjection` owns display timelines. Its ordinary projection
state stores fork metadata and addressed sequence indexes. Large message bodies
live in addressed sequence entries.

`Projection.addressed` is the semantic indexing system. An addressed sequence
index maps item ids, positions, tail windows, range windows, and collection
sentinels onto physical addressed entries. Sequence reads validate item ids and
offsets against the entry they read.

`ProjectionConsumer` is the runtime consumer boundary. A runtime consumer
streams the result of an ordinary `Effect` program that reads projection state
and addressed entries. Event-core runs the effect, tracks those reads, pins the
physical entries that were read, replaces old pins only after new pins are
acquired, reruns the effect when tracked dependencies change, emits the new
materialized value, and releases pins on close.

`DisplayViewRuntime` owns open display views. For each view it stores the
requested shape, the latest accepted snapshot, a subscriber pubsub, and one
scoped `ProjectionConsumer`. It is runtime state, not a projection and not
event-log state.

There is no `DisplayViewProjection`. Display view shape and close are not app
events.

## Data Terms

Display intent is client UI intent: selected session, visible worker stack,
root timeline window, pagination, and presentation mode.

Requested shape is the `DisplayViewShape` derived from display intent. It names
the timeline windows the client wants materialized.

Accepted shape is the agent's answer for a snapshot. It contains only the
parts of requested shape that currently resolve against projection state.

Accepted snapshot is `{ shape, state }`. UI renders accepted snapshots, not
requested shape.

A view id names one logical display view for a session. Multiple stream
subscribers attach to the same logical view and observe the same accepted
snapshot stream.

## Runtime Consumer

Display snapshot materialization is described as an ordinary effect:

```ts
const snapshot = yield* buildDisplayViewSnapshot(shape).pipe(
  ProjectionConsumer.provide(view.consumer),
)
```

Long-lived display views use the stream form:

```ts
const snapshots = ProjectionConsumer.stream(view.consumer)(
  buildDisplayViewSnapshot(shape),
)
```

Inside that effect, display code reads projections through the consumer:

```ts
const timeline = yield* ProjectionConsumer.read(DisplayTimelineProjection)
const fork = timeline.state.forks.get(forkId)
const messages = timeline.addressed.forFork(forkId).messages
const window = messages.resolveTailWindow(fork.messages, limit)
const visibleMessages = yield* messages.readWindow(window)
```

This uses the existing addressed sequence index. Display code chooses windows
from display shape, but it does not invent physical addresses, owner strings,
or pin sets.

The consumer records:

- ordinary projection state reads;
- addressed entry reads;
- addressed collection sentinels touched by window operations.

When materialization succeeds, event-core replaces the consumer's pins with the
new read set and emits the materialized snapshot. If materialization fails,
existing pins remain in place. When tracked ordinary state or addressed content
changes, event-core reruns the effect and emits the new snapshot.

One display view owns one consumer. Independent pin lifetimes use independent
consumers.

## Shape Protocol

The protocol shape API is explicit:

```ts
SetDisplayViewShape({ sessionId, viewId, shape })
StreamDisplayView({ sessionId, viewId })
ResyncDisplayView({ sessionId, viewId })
CloseDisplayView({ sessionId, viewId })
```

`SetDisplayViewShape` is the only shape writer. It creates or updates the
runtime logical view and materializes a new accepted snapshot.

`StreamDisplayView` is read-only. It subscribes to accepted display events for
an existing runtime view. It has no shape field and never changes shape.

`ResyncDisplayView` emits the latest accepted snapshot as a full state event.
It does not negotiate shape.

`CloseDisplayView` closes the runtime view and releases the consumer.

## End-To-End Flow

1. UI actions update display intent.
2. The display controller derives the latest requested `DisplayViewShape`.
3. The controller sends `SetDisplayViewShape`.
4. The controller opens `StreamDisplayView`.
5. ACN resolves the session runtime and relays the command or stream request.
6. `DisplayViewRuntime` starts or updates a `ProjectionConsumer.stream` for the
   requested shape.
7. Event-core tracks projection/addressed reads, replaces the view's pins, and
   emits accepted snapshots.
8. The runtime publishes accepted snapshots to subscribers.
9. ACN converts accepted snapshots to full state events or JSON patches.
10. The SDK filters transport concerns and delivers stream events.
11. The client store applies accepted state/patch events with reference
    preservation.
12. UI derives rendering from intent plus accepted truth.

## Layer Roles

The client controller owns display intent and the selected view lifecycle. It
is the only client-side caller that changes display shape.

The client store owns accepted snapshots. Reference preservation reduces
rerenders by reusing unchanged object identities. It does not preserve hidden
workers as truth and does not mutate requested shape.

The SDK owns typed RPC access, daemon discovery, heartbeat filtering, and
transport recovery. It does not decide worker visibility or synthesize shape.

ACN is a relay and resource owner. It owns stream registrations,
subscriber ref counts, stream sharing, snapshot/patch conversion, and
introspection. It does not infer timeline availability and does not update
shape from `StreamDisplayView`.

The agent owns accepted display truth through `DisplayViewRuntime`. It resolves
requested shape against current projection state, materializes accepted
snapshots, publishes accepted snapshots, and releases runtime consumers.

Event-core owns projection execution, addressed collection semantics,
residency, pin replacement, release, dependency tracking, and materialized
value streaming.

## Addressed Residency

Addressed residency follows accepted windows.

The ordinary projection index locates entries by semantic collection structure:
item ids, positions, tail windows, range windows, and collection sentinels.

Pins claim physical addresses. A display view pins exactly the addressed
entries read while materializing its accepted snapshot.

Pin replacement acquires newly read entries before releasing old entries. This
keeps moving windows resident across shape changes and prevents partial
accepted snapshots from dropping the previous working set on failure.

Producer pins are separate runtime claims owned by active writers such as
streaming assistant messages, tools, thinking messages, and worker
communications.

Residency is not UI retention. Hidden workers are not kept alive for rerender
stability. Reference preservation handles rerender stability; shape
negotiation and consumer pins handle loaded data.

## Worker Navigation

Opening a worker changes client intent. The requested shape includes root plus
the visible worker chain.

Backing out changes client intent. The requested shape removes the hidden
worker unless product policy explicitly keeps it visible.

If a requested worker fork does not exist, accepted shape omits that timeline.
The UI treats desired-but-unaccepted timelines as pending.

## Race-Free Invariants

There is exactly one semantic writer of client display intent: the display
controller.

There is exactly one semantic owner of accepted display truth:
`DisplayViewRuntime`.

Only `SetDisplayViewShape` mutates requested shape.

`StreamDisplayView` never carries shape and never mutates requested shape.

Display view open, shape change, resync, and close do not append app events.

ACN stream sharing is actual hot sharing. A subscriber attach cannot reexecute
a shape-changing open effect.

Accepted-shape mismatch is not a race. It is normal for requested shape to
include a worker that accepted shape temporarily omits.

Resync repairs patch-base drift. It does not chase missing timelines.

Addressed residency follows accepted windows. Pins are backend residency
claims, not UI caches.

## Diagnostic Ladder

Use the earliest layer where state is already wrong.

If `SetDisplayViewShape` commands alternate between root-only and
root-plus-worker, the bug is before or at the agent command boundary: client
controller, SDK recovery, or ACN dispatch.

If requested shapes are stable but accepted snapshots omit the worker, inspect
agent inputs: fork existence, timeline indexes, addressed window resolution,
addressed store reads, pins, and projection errors.

If accepted snapshots are stable but the UI flips, inspect client patch
application, store reference preservation, selectors, and render derivations.

If addressed entries fail to load or pins point at missing physical addresses,
that is an addressed projection integrity failure. It surfaces as a backend
error, not as repeated shape resync.

## What Not To Add

Do not add display-view app events.

Do not add a `DisplayViewProjection`.

Do not put shape on `StreamDisplayView`.

Do not let SDK or ACN replay stale shape-bearing requests.

Do not let ACN treat stream subscribe as shape update.

Do not add client-side worker retention caches.

Do not maintain mutable timeline status state. Derive it from intent and
accepted truth.

Do not use repeated resyncs, retained worker lists, or shape resend loops to
compensate for stale shape writes.

Do not call `ProjectionBus.pinAddressedConsumer` from display runtime code.

## Clean Shape

The clean path is:

`UI intent -> display controller -> SetDisplayViewShape -> DisplayViewRuntime -> ProjectionConsumer -> accepted snapshot -> ACN stream -> SDK -> reference-preserving store -> UI derivation`

Race freedom comes from keeping shape writes single-path and read streams
read-only. Residency comes from event-core consumer tracking over the existing
`Projection.addressed` semantic indexes.
