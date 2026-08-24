---
applies_to:
  - packages/acn-protocol/src/transport/subscription-wire.ts
  - packages/acn-protocol/src/schemas/changes.ts
  - packages/acn-protocol/src/boundary/changes.ts
  - packages/acn-protocol/src/boundary/display.ts
  - packages/acn-protocol/src/boundary/acn.ts
  - packages/acn/src/acn-subscriptions.ts
  - packages/acn/src/acn-subscription-protocol.ts
  - packages/acn/src/changes.ts
  - packages/sdk/src/acn-jit/acn-subscription-protocol.ts
  - packages/sdk/src/jit-rpc/recovering-stream-protocol.ts
  - packages/client-common/src/state/changes.ts
  - packages/sdk/src/inference.ts
---

# ACN subscriptions

An ACN subscription is a core `Subscription` or stream-folded `Query` in an ACN boundary group
whose derived RPC streams domain values. Stream status comes from stable
`AcnRpc.operations(AcnBoundary)`
metadata rather than a separate registry. Opening one is caller-owned observation: it never retains ACN or a session
runtime. Domain handlers produce domain values and consumers receive domain values.

## Wire protocol

The ACN subscription wire protocol is a transport concern implemented only by the two protocol
decorators — the ACN server protocol wraps each encoded domain value in a `payload` frame and
interleaves controls; the SDK client protocol consumes controls and unwraps payloads before Effect
RPC decodes them. Definitions carry no framing and no annotation.

| Frame | Meaning |
| --- | --- |
| `payload` | One encoded domain value |
| `keepalive` | Transport remains live |
| `terminated` | ACN relinquishes the subscription during shutdown |

A quiet domain is normal; absence of both payload and keepalive beyond the liveness interval is
concrete transport failure. Valid `terminated`, transport failure, malformed framing, liveness
failure, and a stream ending without terminal control invalidate observation and enter client
recovery. Recovery selects again, reopens, and rereads authoritative state. Domain failure remains
a domain result.

Client interruption removes only that exact observer; there is no separate close RPC. Shutdown
atomically marks the registry terminal and detaches all registrations before external effects,
interrupts keepalives, attempts terminal frames concurrently under one shared short bound, and
closes every transport regardless of write outcome. No emit, close, interruption, or finalizer is
awaited under registry synchronization. Registration after terminalization receives a closed
transport.

## Change notifications

Connection-global change notifications share one subscription, `StreamChanges`. Each event is a
poke in the clients' query-identity space — `{ query, key?, revision? }` naming the query whose
authoritative data may have changed. The ACN change registry multiplexes every source: a versioned
snapshot commit names its own query with its revision; a storage commit names every query it backs.
Multiplexing neither combines state nor creates a cross-domain authority.

The client drains `StreamChanges` once per connection and invalidates the named query; every
(re)connection after the first invalidates everything, since pokes may have been missed. Nothing
else is derived from pokes and no domain code interprets them.

Native inference uses ICN's `/api/v1/events` SSE stream rather than ACN RPC framing. The authored
Inference Effect Query layer drains that one multiplexed stream and invalidates Hardware, Models,
Packages, Downloads, Instances, and residency-policy Queries. Both drains feed the same
connection-scoped query cache and use the same reconnect-then-reread rule; they do not combine the
underlying authorities or introduce another frontend state system.

## Keyed subscriptions

`WatchFile`, `WatchProjectFiles`, and `StreamDisplayView` carry a resource in their payload. Their
relationship to client state is ordinary code in the owning client service: watches invalidate the
queries they keep fresh and are open exactly while those queries are observed; the display view
subscription is consumed by the display controller.

## Guarantees

- Controls never leak into domain values; definitions carry no framing.
- A quiet live subscription cannot appear complete.
- Reconnection obtains current truth rather than replaying controls as history.
- One observer's cancellation cannot affect another observer or domain state.
- Connection-global invalidations consume one transport subscription without mixing cache authority.
- Subscriber backpressure cannot retain ACN shutdown.
