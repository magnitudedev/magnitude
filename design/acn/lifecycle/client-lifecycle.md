---
applies_to:
  - packages/sdk/src/acn-jit/**
  - packages/sdk/src/jit-rpc/**
  - packages/client-common/src/state/acn-lifecycle.ts
  - cli/src/features/app-shell/**
  - cli/src/platform/**
  - desktop/src/*.ts
  - web/src/platform/**
  - web/scripts/dev-server.ts
---

# ACN client lifecycle

Each client process has one `AcnJitRuntime`. It owns the client's effective ACN identity, selected
exact instance, JIT ensurance, transport recovery, and bootstrap presentation. RPC consumers and UI
surfaces observe that authority; they do not maintain independent identities, endpoint caches,
launch decisions, or recovery loops.

```text
Checking -> Starting / Installing -> Ready(instance)
               |                         |
               +------> Failed <---------+
                           |
                           +-> retry through the same ensure operation
```

These are client presentation states, not another ACN service lifecycle. CLI performs local process
management directly. Desktop and web cross their existing host boundary while preserving the same
typed `AcnProcessManager` behavior.

## ACN association

The runtime owns one association:

```text
AcnAssociation
  identity    ACN version this client now corresponds to
  selected    optional exact Ready ACN instance
```

ACN version is the ACN identity. The association initializes from the client's bundled ACN version,
but that value is only an initializer.

Observing a usable newer ACN in `Starting` or `Ready` advances `identity` immediately and
permanently for that client process. Every later compatibility check, artifact lookup, ensure,
recovery, and launch reads the association. Losing `selected` never regresses `identity`; an
unavailable adopted artifact fails explicitly rather than falling back to an older ACN.

Identity and instance selection update through the same serialized reconciliation. The runtime
exposes identity through one read-only contract; there is no identity constant, header, transport
side channel, or second store consulted after initialization.

## Ensurance and recovery

Initial startup, retry, and concrete transport recovery all enter the same serialized ensure path
defined by [JIT ACN ensurance and upgrades](./jit-spawning.md).

On recovery the runtime rereads current ACN assignment:

- the same exact `Ready` ACN means the request failed locally and does not authorize replacement;
- a newer usable `Starting` or `Ready` ACN advances the association before further action;
- an exact `Starting` ACN is observed within its progress-aware startup window;
- absence enters the two-second publication grace and then delegates to `AcnProcessManager.launch`;
  and
- a stalled or stopping exact instance is passed to `launch` as the replacement target.

The runtime never kills an ACN directly. `AcnProcessManager` joins or advances the durable assignment
change and owns exact cleanup, spawning, and takeover.

Domain failure and caller cancellation do not trigger ACN recovery. Operation duration is not a
transport deadline. Queries may be replayed and observations reopened according to their domain
contract; mutations require durable idempotency or unknown-outcome reconciliation. See
[operation ownership](../../architecture/operation-ownership.md).

Every application RPC is bound to the selected exact ACN instance ID as well as its URL. Direct and
proxied transports carry that ID to ACN dispatch, which rejects a request addressed to another
occurrence. Desktop and web preserve the same typed process-manager errors as local execution;
transport boundaries do not reduce coordination failures to strings.

## Guarantees

- One runtime supplies every RPC consumer with the client's current ACN identity and exact instance.
- Identity never regresses during a client process lifetime.
- A client that adopts a newer ACN can never later launch its bundled older version.
- Initial selection and recovery use one ensure policy.
- The two-second publication grace never authorizes an unfenced spawn.
- Connection loss, observation failure, and application latency do not independently authorize
  replacement.
- Intentional replacement is not reported as a crash; child logs remain diagnostics.
