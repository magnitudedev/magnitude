# Display Views

A display view is a client's windowed subscription to a session's display
state. The agent never streams full display state; it materializes exactly what
an open view's shape requests, and clients grow or move their windows instead
of paginating locally over complete timelines.

## Shape

A view's shape declares which fork timelines the client wants and, for each,
a window: a tail of the most recent N messages or an explicit range, plus
whether the window is live. Everything outside timelines — session identity,
agent statuses, task rows — is always included whole; only timelines are
windowed, because only timelines grow without bound.

The shape belongs to the open view. It is carried in the request that opens the
stream, and shape updates apply only to open views. There is no ambient
per-view configuration anywhere: a view that is not open has no shape, and a
reconnecting client re-establishes its view entirely from the shape it sends at
open. Two subscribers sharing a view id share one view; the most recent shape
wins.

## Requested Versus Accepted

The agent resolves the requested shape against what actually exists. Timelines
requested for forks that do not exist are dropped, and the result is the
accepted shape. Every emission carries the accepted shape alongside the state,
so the client always knows what its window actually covers. Clients must treat
the accepted shape, not their request, as the truth of what they are seeing —
a requested worker timeline may appear in a later emission once that fork
exists.

## Windows, Pins, And Invalidation

Resolving a shape against a timeline's addressed index yields windows over
physical entries. The view pins the physical entries its accepted windows need,
which holds them resident for materialization; releasing or moving the view
releases the pins. This is the view side of the residency model described in
the addressed-state doc.

Live windows subscribe to changed-address notifications for the entries they
cover and re-emit when content changes. Non-live windows are frozen reads:
because old physical entries are retained across structural rewrites, a frozen
scrollback window remains readable and stable no matter how the timeline moves
underneath it.

A view re-resolves its windows when the timeline's structure changes or its
shape changes. Emissions distinguish content updates from shape changes: a
shape change forces a full-state emission, while content changes within an
unchanged shape may be emitted incrementally.

## Transport

The ACN relays a view stream between the agent and the client. It mirrors the
agent's accepted snapshots and normalizes them for the wire: full state on
open, on shape change, and on explicit resync; JSON patches against the
previously sent state otherwise. The relay holds no view semantics of its own —
it reference-counts subscribers per view, forwards shape updates to open views,
and closes the agent-side view when the last subscriber detaches or the view is
explicitly closed.

## Client Pagination

Clients page by reshaping, not by slicing. A terminal client renders the
accepted tail window and, when the user scrolls back, requests a larger tail
limit; expanding a worker timeline adds that fork's timeline to the shape.
"How much history is loaded" is therefore a property of the view's shape on the
agent side, not of client-side buffering — the client never holds more timeline
than its accepted shape describes.

## Snapshots For Inactive Views

A one-shot snapshot of a view that is not open materializes against a default
shape, acquiring and releasing pins around the read. It does not create or
configure a view.
