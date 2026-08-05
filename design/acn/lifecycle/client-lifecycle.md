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

Each client has one process-local authority for bootstrap presentation, its effective ACN target,
selected endpoint, and transport recovery. Screens and RPC consumers observe it; they do not own
independent endpoint caches, retry loops, or launch policy.

```text
Checking -> Starting / Installing -> Ready(endpoint)
               |                         |
               +------> Failed <---------+
                           |
                           +-> retry through the same JIT operation
```

These are client states, not another copy of ACN service lifecycle. Local CLI performs JIT
ensurance directly. Desktop renderer and browser use their existing host process for filesystem and
spawn mechanics, but selection and recovery semantics remain identical.

## Effective target and selection

The lifecycle owns three distinct values:

```text
bundledTarget     target initially requested by this client build
effectiveTarget   highest compatible target adopted by this client
selectedEndpoint  exact ACN currently used
```

`effectiveTarget` begins at `bundledTarget`. Observing a compatible higher target in `Starting` or
`Ready` advances it immediately under the lifecycle's selection lock. It never decreases. Every
later compatibility check, artifact resolution, launch request, and recovery uses the effective
target rather than a compile-time SDK version.

A compatible newer ACN therefore upgrades an older open client without restarting it. If that ACN
later fails, the older client may ensure the adopted target but cannot start its bundled older ACN.
An adopted target that cannot be reproduced locally produces a typed launch failure; it never
falls back to a lower target.

The selected endpoint carries exact process identity and remains bound through desktop and web
proxies. Proxies do not cache a logically independent ACN URL.

## Startup and recovery

Initial preparation and recovery both enter the JIT ensurance defined by
[JIT ACN ensurance and upgrades](./jit-spawning.md). Retry re-enters that same operation rather than
creating a second startup path.

On concrete transport failure or authoritative retirement, the client rereads ACN state:

- the same exact `Ready` ACN means the request failed locally; it is not rejected globally;
- a compatible `Starting` ACN satisfying the effective target enters its 30-second stall window,
  advancing the target first when it is higher;
- a compatible `Ready` successor advances the target and becomes selected;
- no usable registration enters the 2-second publication grace before spawn contention; and
- expiry of a startup window requests coordinated takeover and exact cleanup, not a direct spawn.

The two-second grace covers only an in-flight publication gap. The 30-second window bounds lack of
forward progress by one exact `Starting` ACN. Their complete ownership and expiry semantics are
normative in the JIT design and must not be recreated in UI or transport code.

Domain failure and caller cancellation never trigger ACN recovery. Operation duration has no
transport deadline: a request remains bound until it completes, is canceled, concretely loses
transport, or receives authoritative retirement.

Queries may be replayed and observations reopened according to their domain contract. Idempotent
mutations retry only with durable identity; an ambiguous mutation outcome is reconciled rather than
blindly replayed against a successor. See
[operation ownership](../../architecture/operation-ownership.md).

## Failure semantics

Typed outcomes distinguish artifact acquisition failure, unreproducible adopted target, candidate
startup failure, blocked exact cleanup, local transport failure, RPC incompatibility, and ambiguous
mutation. Incidental child logs remain diagnostics. Intentional replacement is not a crash.

## Guarantees

- All RPC consumers in one client share one effective target, endpoint, and recovery authority.
- Accepting a newer compatible ACN permanently prevents that client from launching an older one.
- `Starting` beyond two seconds does not fall through to an uncoordinated launch.
- Startup and cleanup deadlines retain the meanings defined by JIT ensurance.
- Connection loss never becomes empty authoritative product state.
- Application operation latency cannot move the client away from a live ACN.
- Recovery preserves query, observation, and mutation semantics.
