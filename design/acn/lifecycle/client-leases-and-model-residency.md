---
applies_to:
  - packages/acn/src/client-lease-*.ts
  - packages/acn/src/model-residency-policy.ts
  - packages/acn-protocol/src/boundary/client-lease.ts
  - packages/acn-protocol/src/schemas/client-lease.ts
  - packages/sdk/src/acn-jit/**
  - packages/client-common/src/utils/cli-exit-notice.ts
  - cli/src/**
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
  - packages/icn-protocol/**
---

# Client leases and local-model residency

One interactive client lifetime owns one `ClientId` and one renewable `ClientLease`. Client
presence is an explicit liveness system; RPC activity, subscriptions, inference, and UI state are
not substitutes for it.

## Client lifetime

`AcnJitRuntime` creates one random `ClientId`, a dedicated lease protocol, and one inert scoped
lease owner. Before its first `AcnInstance<AcnReady>` is published, the runtime establishes the
lease through that exact endpoint under the same close/selection admission boundary; renewal then
runs every 15 seconds.
The lease is not an initial ACN bootstrap trigger. `RpcClient.Protocol` is single-consumer, so the lease
client must not share a protocol instance with application RPC clients. The protocol instances do
share the runtime's endpoint-selection and recovery authority. A graceful close releases the lease
explicitly and returns the connected count after removal. Abrupt close relies on expiry.
Graceful release addresses only the exact ACN already selected by the runtime. Closing and scope
finalization never use the recovering protocol to ensure or launch an ACN.

The ACN accepts a renewal for 35 seconds. This tolerates one missed 15-second heartbeat plus
scheduling and transport jitter, but not two full missed heartbeat intervals. Renewal is idempotent
by `ClientId`; release of an unknown ID also succeeds.

## ACN authority

`ClientLeaseManager` is the sole authority for the ephemeral client set, expiry deadlines, and
connected count. One supervisor sleeps to the nearest monotonic deadline and is woken by state
revision changes. Exact renewal generations fence stale expiry work; no heartbeat creates its own
timer fiber.

The first lease publishes connected model-residency policy. The final release or expiry publishes
disconnected policy and commits the empty set. These transitions are serialized. The bounded policy operation runs in
an explicitly interruptible child fiber so its timeout remains effective; the serialized mutation
joins that child uninterruptibly before the matching state commit. Caller cancellation therefore
cannot split policy acknowledgement from its commit. Definite policy failure fails closed by
stopping ACN rather than committing mismatched state.

Client absence never retires ACN. The per-user service owns ACN process lifetime; explicit service
stop or fatal failure ends it. JIT clients coordinate with that same authority rather than creating
a second lifecycle.

## ICN authority

ICN owns physical model residency. ACN sends a generation-fenced policy through the generated ICN
API:

- one or more connected clients: release after 60 minutes continuously idle;
- no connected clients: release after 10 minutes continuously idle.

A newer policy starts a fresh idle interval for a Ready instance with no inference leases.
Exact retries are idempotent. Older generations and equal generations with different content are
rejected. Before each ACN transition, ACN reads the authoritative ICN policy, advances that
generation, writes the desired timeout, and rereads to verify acknowledgement. It therefore remains
correct if another management-API consumer advanced the policy generation. The residency actor
qualifies each idle deadline by exact instance ID and an actor-owned idle generation. Lease
acquisition, lease release, or policy change advances that generation; only a matching deadline
with zero leases may begin idle drain.

If ACN cannot establish a first/final-client policy after bounded retries, it fails closed instead
of committing a client count paired with an unproved model timeout.

## Client presentation

Graceful close releases the lease and returns the post-release connected count. It does not read
Slot residency because Slots contain no runtime state and foreground exit does not own service or
model lifetime. Unknown close observation may show bounded generic service guidance.

## Conformance

- One runtime produces one client identity and one heartbeat schedule.
- Every `RpcClient` owns a distinct single-consumer protocol receiver; those receivers share only
  the runtime's endpoint-selection and recovery authority.
- Graceful close stops renewal and uses a non-recovering protocol bound to the selected exact ACN;
  abrupt scope finalization closes the runtime, stops renewal, and relies on lease expiry.
- Heartbeat and release RPC duration is process-lifecycle neutral.
- Lease expiry uses monotonic time and exact renewal generations.
- First/final transitions alone change ICN residency policy.
- Every disconnected transition gives an idle resident model a fresh 10-minute interval.
- A stale timer or policy message cannot release a newer model or extend its residency.
- Foreground client exit never stops the per-user service or infers residency from Slots.
