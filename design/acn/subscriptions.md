---
applies_to:
  - packages/acn-protocol/src/rpcs/subscription.ts
  - packages/acn-protocol/src/schemas/subscription.ts
  - packages/acn/src/acn-subscriptions.ts
  - packages/acn/src/acn-subscription-protocol.ts
  - packages/sdk/src/acn-jit/acn-subscription-protocol.ts
  - packages/sdk/src/jit-rpc/recovering-stream-protocol.ts
  - packages/acn/src/display-view-streams.ts
  - packages/acn/src/changes.ts
  - packages/client-common/src/state/changes.ts
  - packages/client-common/src/state/agent-client.ts
  - packages/acn-protocol/src/rpcs/changes.ts
  - packages/acn-protocol/src/schemas/changes.ts
---

# ACN subscriptions

An ACN subscription is caller-owned observation. Opening one never starts product work or retains
ACN or a session runtime. Domain handlers and consumers see only payload values; framing remains a
transport concern.

Connection-global change notifications share one subscription, `StreamChanges`. Each event is a
poke in the clients' query-identity space — `{ query, key?, revision? }` naming the query whose
authoritative data may have changed — so clients invalidate that query and nothing else is derived
from it. A mirror commit names its own Get RPC tag with its revision; a store commit names every
query it backs. Multiplexing notifications neither combines their state nor creates a cross-domain
authority.

| Frame | Meaning |
| --- | --- |
| `payload` | Domain observation |
| `keepalive` | Transport remains live |
| `suspended` | Session runtime unloaded; subscription remains open |
| `terminated` | ACN relinquishes the subscription during shutdown |

Keepalives are consumed below domain streams. A quiet domain is normal; absence of both payload and
keepalive beyond the liveness interval is concrete transport failure.

Session suspension preserves the client's last accepted snapshot. Later materialization, shape
change, or resynchronization reloads the runtime, attaches a new generation, and sends a complete
snapshot. Suspension neither closes the stream nor initiates ACN recovery.

Valid `terminated`, transport failure, malformed framing, liveness failure, and a stream ending
without terminal control invalidate observation and enter client recovery. Recovery selects again,
reopens, and rereads authoritative state. Domain failure remains a domain result.

Client interruption removes only that exact observer; there is no separate close RPC. Closing the
last display subscription removes its display registration.

Shutdown atomically marks the registry terminal and detaches all registrations before external
effects. It then interrupts keepalives, attempts terminal frames concurrently under one shared
short bound, and closes every transport regardless of write outcome. No emit, close, interruption,
or finalizer is awaited under registry synchronization. Registration after terminalization receives
a closed transport. Session suspension follows the same lock/delivery discipline without closing.

## Guarantees

- Controls never leak into domain values.
- A quiet live subscription cannot appear complete.
- Session unload preserves observation and last accepted state.
- Reconnection obtains current truth rather than replaying controls as history.
- One observer's cancellation cannot affect another observer or domain state.
- Connection-global invalidations consume one transport subscription without mixing cache authority.
- Subscriber backpressure cannot retain ACN shutdown.
