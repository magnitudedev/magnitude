---
applies_to:
  - packages/sdk/src/acn-jit/acn-recovering-client.ts
  - packages/sdk/src/acn-jit/lifecycle.ts
  - packages/sdk/src/jit-rpc/**
  - packages/client-common/src/state/acn-lifecycle.ts
  - cli/src/features/app-shell/**
  - cli/src/platform/**
  - desktop/src/*.ts
  - web/src/platform/**
  - web/scripts/dev-server.ts
---

# ACN client lifecycle

Each client has one authority for bootstrap state, its selected ACN endpoint, and transport
recovery. Screens and RPC consumers observe that authority; they do not own endpoint caches,
discovery, retry, or replacement policy.

```text
Checking -> Starting / Installing -> Ready(endpoint)
               |                         |
               +------> Failed <---------+
                           |
                           +-> retry through the same authority
```

These are client states, not another copy of ACN service lifecycle. `Installing` is presentation of
host preparation. `Ready` retains the exact endpoint identity and fencing epoch selected by the host
coordinator.

## Host boundary and selection

The client asks the host coordinator defined by [JIT spawning and handoff](./jit-spawning.md) to
observe or ensure an endpoint satisfying explicit
coordination, RPC, storage, and release requirements. Local CLI, desktop main, and web host adapters
transport the same typed contract; browser and renderer processes never implement selection policy.

The selected `{ instanceId, epoch, endpoint }` is used by every RPC consumer and remains bound
through proxies. A proxy cannot cache another process URL, and every dispatch carries the expected
identity and epoch so port reuse or stale routing cannot reach a different ACN.

Initial preparation may begin before rendering. CLI can wait for the first observable startup state
to avoid a blank frame; other clients may render immediately. Retry re-enters this same lifecycle
and coordinator operation rather than creating a second startup path.

## Recovery

A request failure is evidence about that request, not permission to replace ACN. Domain failure and
caller cancellation never trigger recovery. Operation duration has no transport deadline: a request
remains bound until it completes, is canceled, concretely loses transport, or receives authoritative
retirement.

On concrete transport failure or retirement, the client rereads coordinator state:

- the same exact `Ready` ACN means the request failed locally; it is not rejected globally;
- a compatible `Starting` successor is observed without spawning another candidate;
- a compatible `Ready` successor becomes the selected endpoint; and
- absence or incompatibility is returned to coordinator `ensure`, which independently decides
  whether mutation is legal.

Connection refusal/reset, exact process exit, malformed or prematurely ended protocol, and valid
terminal control are concrete recovery evidence. Slow response, one health timeout, filesystem
uncertainty, domain error, and cancellation are not.

Queries may be replayed and observations reopened with authoritative reread. Idempotent mutations
may retry only under their domain contract; an ambiguous mutation outcome is reconciled rather than
blindly replayed against a successor. See [operation ownership](../../architecture/operation-ownership.md).

## Failure semantics

Typed client outcomes distinguish host preparation failure, candidate failure, incompatible active
authority, blocked handoff, local transport failure, RPC incompatibility, ambiguous mutation, and
explicit forced-replacement failure. Incidental child log tails remain diagnostics and never choose
the failure class. Intentional supersession is not reported as a crash.

## Guarantees

- All consumers in one client share one selected endpoint and recovery authority.
- Client transport failure cannot authorize machine-wide replacement.
- Connection loss never becomes empty authoritative product state.
- Endpoint identity survives every host and proxy boundary.
- Operation latency cannot move the client away from a live ACN.
- Recovery preserves query, observation, and mutation semantics rather than treating every RPC alike.
