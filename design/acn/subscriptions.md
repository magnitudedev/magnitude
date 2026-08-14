---
applies_to:
  - packages/acn-protocol/src/rpcs/subscription.ts
  - packages/acn-protocol/src/rpcs/stream.ts
  - packages/acn-protocol/src/schemas/subscription.ts
  - packages/acn/src/acn-subscription*.ts
  - packages/acn/src/display-view-streams.ts
  - packages/sdk/src/acn-jit/acn-subscription-protocol.ts
  - packages/sdk/src/jit-rpc/recovering-stream-protocol.ts
  - packages/agent/src/display-view/runtime.ts
  - packages/client-common/src/display-view-controller/controller.ts
  - packages/client-common/src/headless/session-runner.ts
---

# ACN subscriptions

An ACN subscription is caller-owned observation. It is passive by default and never retains ACN or a
session runtime. A subscription payload may explicitly request one finite authoritative
materialization during admission; this can load a session runtime, but it neither starts product work
nor retains the runtime after that query completes. Domain handlers and consumers see only payload
values; framing remains a transport concern.

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
last display subscription removes its display registration. Registration creation, queue
subscription, subscriber-count admission, and finalizer ownership cross no interruptible gap. A
last-subscriber cleanup marks the exact registration closing and detaches it under its short lock,
releases that lock before best-effort runtime close, then reacquires it to remove registry state and
complete teardown even when the external effect fails. A replacement subscriber observes the closing
state, waits for that key-linearized teardown, and retries after removal, so an old close cannot affect
the replacement registration. Runtime retirement and shutdown can still acquire display
serialization while resident close admission is pending.

Encoded transport admission follows the same rule: installing the subscription and recording its
exact unregister handle are one uninterruptible state transition. Cancellation therefore cannot
leave a keepalive registration without an owner. Every encoded RPC request id is reserved before it
is forwarded, including ordinary finite RPCs. Overlapping reuse is malformed and closes that client
after unregistering its subscriptions instead of installing a replacement that a predecessor `Exit`
could remove. The closed client remains poisoned while any forwarded request from that transport is
still active, so callbacks queued before closure cannot reopen it. Each authoritative terminal `Exit`
removes exactly its request occurrence; the final one removes the poison record. Poison bookkeeping
is therefore bounded by active transport lifetimes rather than daemon lifetime. Sequential reuse
after terminal removal is valid, and the RPC server emits no second terminal response for an
occurrence. Runtime-generation reattachment considers only
registrations with a live subscriber, rechecking registry identity and subscriber ownership under
the registration lock immediately before attaching. Busy-runtime acquisition also occurs outside
that lock, so final-subscriber cleanup is not blocked by resident-gate or retirement resolution. A
stale runtime-change scan or interrupted zero-subscriber admission therefore cannot attach later.

Each registration serializes shape admissions independently from its short state lock. One admission
records intent, acquires the runtime, atomically builds the requested shape and full snapshot in the
agent runtime, then commits or rolls back before the next shape admission may record intent. Runtime
retirement can still acquire the short state lock and detach an old generation while acquisition is
pending. After commit, forwarding drops state events whose shape differs from the committed
registration shape. Every publication receives a monotonic registration sequence. A materializing
subscriber drains only records at or below its accepted commit sequence, receives that full state
first, then preserves buffered and live post-fence events in sequence. Failed or cancelled
admission returns to the last committed shape, never another in-flight intent. Passive admissions
never mutate shape intent.

Shutdown atomically marks the registry terminal and detaches all registrations before external
effects. It then interrupts keepalives, attempts terminal frames concurrently under one shared
short bound, and closes every transport regardless of write outcome. No emit, close, interruption,
or finalizer is awaited under registry synchronization. Registration after terminalization receives
a closed transport. Session suspension follows the same lock/delivery discipline without closing.

## Guarantees

- Controls never leak into domain values.
- A quiet live subscription cannot appear complete.
- Session unload preserves observation and last accepted state.
- Explicit admission materialization installs observer cleanup before loading or attaching state.
- A materializing admission refreshes and emits a full state even when the view is already attached
  to the current runtime generation; its accepted full state is always the subscriber's first record.
- Shape mutation and snapshot creation commit atomically; failure leaves the prior runtime view and
  registration shape authoritative.
- Display updates are published through per-subscriber PubSub queues, so overlapping observers each
  receive every authoritative event rather than competing for one queue.
- Cancellation cannot land between registration and cleanup ownership at either the display or
  encoded-transport layer.
- Cleanup removes registry authority even when runtime close or detach fails.
- Reconnection obtains current truth rather than replaying controls as history.
- One observer's cancellation cannot affect another observer or domain state.
- Subscriber backpressure cannot retain ACN shutdown.
