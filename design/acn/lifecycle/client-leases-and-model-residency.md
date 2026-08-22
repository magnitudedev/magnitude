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

The first lease acquires one ACN retention capability and publishes connected model residency. The
final release or expiry publishes disconnected residency, commits the empty set, and releases that
exact retention capability. These transitions are serialized. The bounded policy operation runs in
an explicitly interruptible child fiber so its timeout remains effective; the serialized mutation
joins that child uninterruptibly before the matching state commit. Caller cancellation therefore
cannot split policy acknowledgement from its commit. Definite policy failure fails closed by
stopping ACN rather than committing mismatched state.

Connected clients retain ACN independently of work demand. With no connected client and no work,
ACN begins a fresh 30-minute idle interval.

## ICN authority

ICN owns physical model residency. ACN sends a generation-fenced policy through the generated ICN
API:

- one or more connected clients: release after 60 minutes continuously idle;
- no connected clients: release after 10 minutes continuously idle.

A newer policy starts a fresh idle interval for a resident generation with no inference leases.
Exact retries are idempotent. Older generations and equal generations with different content are
rejected. Idle release revalidates the exact model generation, policy generation, zero inference
leases, and the complete interval under the model controller's mutation authority.

If ACN cannot establish a first/final-client policy after bounded retries, it fails closed instead
of committing a client count paired with an unproved model timeout.

## Client presentation

Graceful CLI exit reads the authoritative model-slot snapshot before releasing its own lease. It
uses the returned post-release client count to report either the fresh 10-minute idle boundary or
the number of other connected clients. `AcnJitRuntime.close` owns that observation and release;
client surfaces do not receive its private lease client or raw lease handle. Only local `Loading`
and `Ready` instances produce a model notice. Distinct instances are all named and duplicate slot
projections are deduplicated. Unknown observation uses bounded fallback copy and never invents a
count or deadline.

## Conformance

- One runtime produces one client identity and one heartbeat schedule.
- Every `RpcClient` owns a distinct single-consumer protocol receiver; those receivers share only
  the runtime's endpoint-selection and recovery authority.
- Graceful close stops renewal and uses a non-recovering protocol bound to the selected exact ACN;
  abrupt scope finalization closes the runtime, stops renewal, and relies on lease expiry.
- Heartbeat and release RPCs are lifecycle-neutral and do not count as work demand.
- Lease expiry uses monotonic time and exact renewal generations.
- First/final transitions alone change ICN residency policy.
- Every disconnected transition gives an idle resident model a fresh 10-minute interval.
- A stale timer or policy message cannot release a newer model or extend its residency.
- CLI notices are derived only from the slot snapshot and post-release count.
