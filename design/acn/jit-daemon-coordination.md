---
applies_to:
  - packages/sdk/src/jit-rpc/**
  - packages/sdk/src/acn-jit/**
  - packages/acn-protocol/src/acn-registry.ts
  - packages/acn-protocol/src/schemas/acn-health.ts
  - cli/src/**
  - desktop/src/**
  - web/src/platform/**
  - web/src/renderer.tsx
  - web/scripts/dev-server.ts
  - scripts/accept-release-candidate.ts
  - packages/acn/src/daemon-lifecycle.ts
  - packages/acn/src/daemon-registration.ts
  - packages/acn/src/machine-ownership.ts
  - packages/acn/src/identity.ts
  - packages/acn/src/server.ts
  - packages/acn/src/service-lifecycle.ts
  - packages/acn/src/peer-acn.ts
  - packages/acn/src/process-identity.ts
  - packages/acn/src/icn/**
---

# ACN process lifecycle and client recovery

One data root has one canonical ACN. Every ACN is visible, every lifecycle is authoritative, and
no cooperative shutdown or child operation may retain process ownership indefinitely.

## Boundaries

```text
client lifecycle            host boundaries                 ACN
----------------            ---------------                 ---
owns selected endpoint      Discovery.current() -> state?   owns service lifecycle
owns recovery policy        Launcher.launch() -> endpoint   owns instance visibility
owns startup presentation   ChildProcessSpawner (private)   owns peer termination
owns transport retries      no selection policy             owns its private ICN
```

The host boundary is two contracts because query and mutation have different meanings:

```ts
DaemonDiscovery.current(): Effect<Option<DaemonStatus>, DaemonError>
DaemonLauncher.launch(command): Stream<DaemonLaunchEvent, DaemonError>
```

`DaemonStatus` is `{ id, version, url, pid, state }`. Discovery reports the canonical daemon
honestly, including `Starting`, `Ready`, or `Stopping`; it does not apply compatibility or selection
policy. Launch performs only launch and streams progress. `ChildProcessSpawner` is the private OS
adapter beneath the local launcher; Electron and Bun supply implementations.

No boundary exposes ensure, replace, invalidate, exclusion, lifecycle lookup, or recovery policy.
Local, remote HTTP, and Electron IPC adapters preserve these two contracts independently.
The web adapter represents the selected ACN through its same-origin server; that server proxies
ACN traffic to the authoritative process rather than exposing the process origin to the browser.

The client `AcnLifecycle` is the sole client-side connection authority. Its `Ready(endpoint)` state
is the selected endpoint; there is no second endpoint cache. `/health` is the sole external ACN
lifecycle query. ACN RPC admission reads the same lifecycle authority that produces `/health`.

## ACN visibility

```text
acn/registry.json                 canonical ACN
acn/instances/<acn-id>.json      every ACN process
```

An instance record contains the opaque ACN ID, PID, host process-start identity, version, and—after
the server binds—its URL. Launch publishes it before canonical registration or
machine ownership. Clean exit removes only that exact record.

Instance enumeration and peer reconciliation are also garbage collection:

- malformed records are removed;
- records whose PID and process-start identity no longer identify the same process are removed;
- live records are never discarded because of age;
- PID alone is never sufficient to signal or remove an instance.

The canonical registration is an atomic pointer containing ID, PID, version, and URL. An ACN that
loses that exact canonical ID enters `Stopping`. Registration is selection; instance records are
visibility. Neither substitutes for the other.

## Starting and replacing ACN

Client policy is:

```text
need endpoint
  -> query canonical ACN
  -> compatible Ready? use it
  -> compatible Starting? reflect state; wait bounded
  -> absent / Stopping / incompatible? launch ACN

selected endpoint fails
  -> query briefly for a different compatible Ready/Starting ACN
  -> found? use it
  -> otherwise launch ACN
```

The failed endpoint is compared only inside the client lifecycle owner. It is not passed into the
platform boundary.

Local `launch` creates a candidate, except that simultaneous launches may join the same
candidate. A bounded filesystem election coalesces concurrent mutations but grants no ownership.
Every contender re-queries after election, so it never acts on stale pre-election state. A stale
election is quarantined after a bounded wait and cannot prevent startup.

The candidate binds its final server, publishes its endpoint, becomes canonical, and removes all
other visible ACNs:

```text
POST /shutdown  (exact target ACN ID)
  -> wait 2s
  -> SIGINT      (only if PID + process-start identity still match)
  -> wait 2s
  -> SIGKILL     (same exact-process check)
  -> bounded proof of exit
```

Canonical ownership is rechecked before peer signals. The predecessor normally observes ownership
loss and stops itself; peer termination is the independent enforcement path. The candidate acquires
machine ownership and starts its private ICN only after peers release ownership.

## Version selection

| Current healthy ACN | Client action |
| --- | --- |
| same release | use current |
| newer compatible release | use current |
| older or incompatible release | discovery still reports it; client launches expected ACN |

Version comparison happens only after health identity matches registration. Starting the expected
ACN, canonical publication, and peer removal form one replacement behavior; “replace” is not a
separate platform operation.

## Lifecycle authorities

```text
client: Checking -> Starting/Installing -> Ready -> Starting ...
                                    \----> Failed -> Starting ...

ACN:    Starting -> Ready -> Stopping

ICN:    Starting -> Ready -> Stopping -> Stopped
```

Use `defineFSM` where these concrete states and transitions exist. State without meaningful
transitions remains ordinary state; it is not forced into an FSM.

Each owner:

- performs all transitions through its FSM;
- derives admission and responses from that same state;
- exposes state through its existing protocol boundary;
- never accepts a mutation that bypasses lifecycle admission.

ACN `Stopping` closes RPC and activity admission immediately. ICN `Stopping` closes operation
admission immediately. New callers therefore receive a terminal typed result instead of joining a
process that is draining.

## Finite failure behavior

Every cross-process phase has a deadline or an independently enforceable fallback:

| Phase | Bound / terminal behavior |
| --- | --- |
| ACN health probe | request timeout; not current |
| spawn election | bounded wait; stale election quarantine |
| candidate publication | deadline; terminate and reap exact candidate |
| ACN ownership acquisition | deadline; candidate stops |
| RPC response start | deadline; recover against a different ACN |
| subscription shutdown writes | detach first; concurrent bounded sends and closes |
| application finalizers | bounded scope close |
| peer shutdown | POST -> SIGINT -> SIGKILL, each bounded |
| ICN child operation during ACN shutdown | bounded application-scope close, then child termination |
| ICN shutdown | SIGTERM, bounded wait, SIGKILL, bounded proof of exit |

Shutdown ordering is:

```text
commit Stopping / close admission
  -> detach subscriptions from authority
  -> best-effort bounded terminal notification
  -> bounded application-scope close
  -> bounded ICN shutdown
  -> process exit and exact instance cleanup
```

No notification write or finalizer is allowed to retain process ownership indefinitely.

## Client behavior

All RPC consumers in one client share one `AcnJitRuntime`, hence one lifecycle authority and one
recovering protocol. Startup UI reads that lifecycle; screens and commands do not implement their
own discovery, retry, or endpoint state.

Transport failure retries finite work once through recovery. Domain failure and caller cancellation
do not trigger ACN recovery. Resident subscriptions recover only after their terminal control or a
liveness/transport failure. Recovery never knowingly selects the exact failed ACN ID.

The CLI may call `prepare` before rendering to avoid a blank bootstrap frame. Other clients may
render immediately and observe the same state. `retry` re-enters selection through the same client
lifecycle owner; it does not create a second path.

## Invariants

- Every live ACN that reached launch publication has one discoverable instance record.
- At most one ACN is canonical; only it may retain machine ownership.
- Every client has one selected endpoint authority.
- Discovery, launch, and child spawning contain no recovery or replacement policy.
- `/health`, RPC admission, and shutdown read one ACN lifecycle authority.
- A failed or hung ACN can delay recovery only by explicit finite bounds.
- ACN shutdown cannot be blocked by client backpressure.
- A hung ICN operation cannot prevent bounded ICN termination or ACN replacement.
