---
applies_to:
  - packages/sdk/src/acn-jit/acn-recovering-client.ts
  - packages/sdk/src/acn-jit/acn-instance-manager.ts
  - packages/sdk/src/acn-jit/local-acn-instance-manager.ts
  - packages/sdk/src/acn-jit/remote-acn-instance-manager.ts
  - packages/sdk/src/jit-rpc/**
  - packages/client-common/src/state/acn-lifecycle.ts
  - cli/src/features/app-shell/**
  - cli/src/platform/**
  - desktop/src/*.ts
  - web/src/platform/**
  - web/scripts/dev-server.ts
---

# ACN client lifecycle

Each interactive client owns one `AcnJitRuntime`. The runtime owns the client's effective ACN
identity, exact selected `AcnInstance<AcnReady>`, single-flight selection, recovering transport,
`ClientId`, `ClientLease`, bootstrap presentation, and one-way close. It does not interpret
coordination state, probe health, choose replacement, or manage processes; those belong to
`AcnInstanceManager`.

```text
Checking -> Starting / Installing -> Ready
               |                     |
               +------> Failed <-----+
                          |
                          +-> explicit retry through AcnInstanceManager
```

These are presentation states, not another ACN service lifecycle.

## Association and selection

```text
AcnAssociation
  identity    monotonic minimum ACN identity
  selected    optional exact AcnInstance<AcnReady>

ActiveSelection
  one shared deferred outcome
```

The association starts at the bundled SDK identity. Only successful ready selection adopts a newer
identity. The durable `AcnRevisionStore` prevents an older client from launching an older ACN
while a newer revision is selected, so waiting until readiness cannot permit downgrade. Losing the
selected endpoint never regresses identity.

Selection is a true single-flight operation. Bootstrap, retry, lease, and application demand share
one scoped owner and one exact outcome while selection is active; a semaphore that merely queues
new operations is insufficient. The owner calls `AcnInstanceManager.ensure`, projects progress into
client presentation, and atomically publishes only terminal `AcnInstance<AcnReady>` values. Every
typed manager terminal failure is projected to `Failed`; explicit retry starts one new ensure
occurrence with a fresh absolute deadline.

Runtime construction explicitly starts initial selection. It also constructs one inert, scoped
lease owner whose renewal fiber is gated by a deferred. Immediate lease establishment, selected
instance publication, and the open/closed check occur in one admission critical section. The
heartbeat starts only after that establishment succeeds.

## Recovery

Every RPC carries both URL and exact ACN instance ID. ACN dispatch rejects another occurrence.
Transport failure clears only the matching failed selection, joins or starts the same selection
single-flight, then retries the exact request according to its transport contract. Domain failure
and caller cancellation do not trigger recovery.

A successful selection is a point-in-time fact. Retirement may begin after selection; exact request
addressing prevents misrouting and recovery handles that unavoidable race. Desktop and web preserve
the same typed ensure stream and cancellation semantics as local execution.

Each concrete `RpcClient` owns its own single-consumer protocol receiver. The private lease client
and application clients share semantic selection/recovery authority, never a protocol receiver.

## Close

Close is one-way and idempotent. It marks the runtime closed under selection admission, closes the
selection scope and awaits interruption, stops heartbeat renewal, then freezes the selected exact
endpoint. Bounded model observation and lease release use a non-recovering protocol bound only to
that endpoint. Close and scope finalization never ensure, discover, replace, or launch an ACN.

Selection publication and lease establishment check `open` under the same admission boundary as
close. If establishment wins, close observes that exact selection and releases its lease; if close
wins, no lease is established or selection published. Each host invokes runtime close before
destroying its scope.
Browser back/forward-cache suspension is not close; lease expiry remains authoritative if renewal
cannot run while suspended.

## Guarantees

- Only `AcnInstance<AcnReady>` enters endpoint selection.
- Identity never regresses during one client lifetime.
- Initial selection, retry, lease recovery, and application recovery share one selection outcome.
- Client presence does not implicitly own bootstrap policy.
- Transitional assignment and temporary health failure cannot independently become startup failure.
- Exact addressing prevents a stale selection from reaching a successor.
- Close cannot publish an endpoint, create a lease, or invoke ensurance after closing begins.
- Intentional replacement is not reported as a crash; process output remains diagnostic.
