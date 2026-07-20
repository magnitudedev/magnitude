---
applies_to:
  - packages/sdk/src/jit-rpc/**
  - packages/sdk/src/acn-jit/**
  - packages/sdk/src/daemon-spawner.ts
  - packages/sdk/src/local-spawner.ts
  - packages/sdk/src/recovering-client.ts
  - packages/sdk/src/remote-spawner.ts
  - cli/src/platform/**
  - desktop/src/platform.ts
  - desktop/src/renderer.tsx
  - web/src/platform/**
  - web/src/renderer.tsx
  - web/scripts/dev-server.ts
  - packages/acn/src/daemon-lifecycle.ts
  - packages/acn/src/daemon-registration.ts
  - packages/acn/src/agent-runtime.ts
---

# JIT ACN coordination and daemon election

Clients reach ACN through one process-local, lazy coordination authority. Constructing that
authority performs no daemon discovery or spawn. The first RPC demand ensures an endpoint, and all
RPC consumers in the application process share that same ensure state even when they build
independent protocol layers or run in independent Effect runtimes.

The coordinator preserves just-in-time daemon lifecycle while preventing AtomRPC, display streams,
status streams, file watches, and other consumers from independently spawning ACNs.

## Ownership

The generic JIT RPC layer owns:

- process-local endpoint caching;
- single-flight discovery and spawn;
- endpoint-specific invalidation after transport failure; and
- reconnecting RPC transport behavior.

The ACN JIT adapter owns:

- construction of one coordinator from the `DaemonSpawner` service;
- ACN RPC path, resident-stream policy, and infrastructure-error mapping; and
- a reusable protocol layer that closes over the already constructed coordinator.

The local daemon spawner owns version-scoped registry discovery, health verification, binary
resolution, cross-process spawn election, and waiting for the canonical winner. Remote spawners
delegate those operations to their authoritative host.

Clients own application composition. Each client platform acquires its spawner and coordinator
once during startup and supplies the resulting reusable protocol layer to every consumer.
`client-common` does not discover, spawn, or elect daemons.

ACN owns its canonical registration, orderly server shutdown, session disposal, and exactly one
private ICN child for the ACN scope.

## Process-local coordinator

The coordinator is an ordinary Effect-created service value. Its mutable state and synchronization
are allocated through Effect primitives; they are not module globals, unsafe constructors, or
mutable closure caches hidden inside a Layer recipe.

Its logical state is:

```text
Unresolved --ensure--> Ensuring --success--> Resolved(endpoint)
     ^                    |                       |
     |                    +--failure-------------+
     |                                            |
     +--------- invalidate matching endpoint ----+
```

Required behavior:

- concurrent ensure callers perform one discovery/spawn sequence;
- successful discovery or spawn is cached before waiters resume;
- a failed ensure restores the unresolved state;
- an invalidation clears only the endpoint used by the failed attempt;
- a delayed failure from an older endpoint cannot clear a newer endpoint;
- domain RPC failures and caller cancellation do not invalidate ACN; and
- coordinator construction itself performs no daemon I/O.

The recovering protocol consumes an existing coordinator. It never constructs or memoizes one.
Independent builds of the reusable protocol layer may allocate request/transport resources, but
they all use the same coordinator value.

Effect Layer memoization is scoped to a Layer build and is not the daemon-sharing mechanism.
Passing the same unbuilt Layer recipe to independent runtimes does not establish shared mutable
state.

## Cross-process election

Separate application processes can observe an absent registration concurrently. Before spawning,
the local spawner acquires an exclusive claim scoped to the expected ACN compatibility/version key.
After acquiring the claim it must re-read and re-probe the registry. If another process published a
healthy compatible ACN while this caller waited, the caller returns that owner without spawning.

The winner holds the claim until a canonical healthy registration is observable or startup fails.
Other contenders wait for the claim and then repeat the mandatory health recheck. They do not spawn
speculative candidates.

The claim uses an atomic filesystem operation and has a bounded stale-claim recovery policy. Claim
release and stale recovery operate only on the exact per-version claim path. A claim contains no
credentials or session data.

A healthy compatible registration is never intentionally replaced by a new candidate. ACN's
ownership watchdog is defensive corruption/failure detection, not the election algorithm.

## Recovery

RPC transport failure invalidates the endpoint used by that attempt. The next ensure first performs
authoritative discovery. It reuses a healthy registered ACN when available and enters election only
when no healthy compatible owner exists.

Resident streams reconnect independently through the coordinator. Sharing coordination does not
merge streams, pin one HTTP connection, serialize RPC traffic, or make ACN eager.

## Shutdown

BunRuntime signal handling interrupts the root Effect fiber. Normal SIGTERM/SIGINT and recoverable
ownership loss therefore unwind the ACN Layer scope.

Agent runtime scope finalization disposes all live sessions. ICN lifecycle finalization cancels its
requests, sends bounded graceful termination, escalates if required, and reaps the child. ACN must
not call `process.exit` on a normal or recoverable ownership-loss path because immediate process
exit skips those finalizers.

ICN parent-liveness monitoring remains a crash backstop for an ACN that cannot run cleanup. It is
not a substitute for scoped shutdown.

## Acceptance criteria

- Mounting every RPC consumer in one client process against an empty registry spawns one ACN.
- Independently built protocol layers from one platform perform one shared discovery/spawn.
- Concurrent client processes for one compatibility key spawn at most one winning candidate and
  converge on its registration.
- A contender that waited for election rechecks health and does not replace the winner.
- Killing the resolved ACN causes one coordinated recovery and a stable replacement endpoint.
- A healthy ACN is never replaced merely because another caller observed stale registry state.
- SIGTERM, SIGINT, idle shutdown, and ownership-loss shutdown unwind ACN scope and reap its ICN.
- One ACN owns one ICN throughout its ready lifetime.
