# Display Architecture

Display is a shape-keyed, windowed observation of event-sourced session state.
The client owns display intent, the agent owns accepted display truth, and the
ACN boundary transports state and patch events between them.

## Boundary

`Display.StreamDisplayView` is one ACN subscription whose input is
`{ sessionId, shape }`. Shape is part of subscription identity: changing shape
means observing a different subscription. Opening or reopening a subscription
always starts with a complete state event, followed by patch events.

There is no separate shape command, resync command, close command, or logical
view registry in the wire contract. Effect Query owns keyed subscription
lifetime; closing the final observer releases the server stream after its
retention period.

## Client ownership

The display controller derives `DisplayViewShape` from selected session,
visible worker stack, timeline windows, pagination, and presentation mode. It
observes `Display.StreamDisplayView` for that exact shape.

The display store applies complete state and patch events while preserving
unchanged object identities. A fresh state event resets the patch base after a
new subscription or reconnect. The store does not copy hidden server truth or
drive transport lifetime independently of the controller.

## Server ownership

ACN resolves the requested session runtime and shares streams for identical
`sessionId` and shape inputs. It converts accepted snapshots into one complete
state event followed by patches. The agent materializes each accepted snapshot
from projection state and addressed entries; backend residency follows the
entries read for the requested windows.

## Flow

```text
UI intent
  -> display controller derives shape
  -> Display.StreamDisplayView({ sessionId, shape })
  -> ACN shared display stream
  -> agent projection materialization
  -> state / patch events
  -> reference-preserving client store
  -> UI derivation
```

## Invariants

- The display controller is the only semantic writer of display intent.
- Subscription input is the complete observation argument.
- Every subscription generation begins with a complete state event.
- Patch events apply only to the current generation's state base.
- ACN and SDK do not infer worker visibility or rewrite shape.
- Hidden timelines are not retained as client-authored server truth.
- Backend addressed residency is separate from UI reference preservation.

## Diagnosis

If requested shape is wrong, inspect UI intent and controller derivation. If
shape is correct but accepted state omits data, inspect agent projections,
addressed reads, and session runtime state. If accepted events are correct but
rendering is wrong, inspect patch application, selectors, and UI derivation.
