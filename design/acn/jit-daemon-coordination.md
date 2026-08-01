---
applies_to:
  - packages/sdk/src/jit-rpc/**
  - packages/sdk/src/acn-jit/**
  - packages/sdk/src/daemon-spawner.ts
  - packages/sdk/src/local-spawner.ts
  - packages/sdk/src/remote-spawner.ts
  - packages/sdk/src/binary.ts
  - packages/acn-protocol/src/acn-registry.ts
  - packages/acn-protocol/src/schemas/acn-health.ts
  - cli/src/**
  - desktop/src/platform.ts
  - desktop/src/main.ts
  - desktop/src/preload.ts
  - desktop/src/desktop-rpc.ts
  - desktop/src/renderer.tsx
  - web/src/platform/**
  - web/src/renderer.tsx
  - web/scripts/dev-server.ts
  - packages/acn/src/daemon-lifecycle.ts
  - packages/acn/src/daemon-registration.ts
  - packages/acn/src/machine-ownership.ts
  - packages/acn/src/identity.ts
  - packages/acn/src/server.ts
  - packages/acn/src/startup-state.ts
  - packages/acn/src/icn/**
---

# Shared ACN startup and upgrades

All clients using the same data root share one ACN. That ACN owns one private ICN.

## Connecting from a client

Each client process creates one `AcnJitRuntime`. All RPC consumers share its protocol layer,
coordinator, lifecycle state, and cached endpoint. Runtime construction itself does not start ACN.

The runtime exposes lifecycle control to clients only through its narrow `AcnStartup` capability:
read-only state, pre-render `prepare`, and `retry`. The SDK owns discovery, cold-start handoff,
single-flight ensure, and retry coordination. `Platform` does not expose the complete runtime.
client-common observes the SDK stream through a private read-only Effect Atom; it does not copy
daemon state into writable client state.

Cached endpoints are generation-specific. A failed request can invalidate only the endpoint it used,
not a newer endpoint discovered by another request. Domain errors and caller cancellation do not
invalidate ACN.

## Finding the current ACN

The data root contains one canonical registration with ACN identity, version, URL, and PID.
Registration is atomically published as mode `0600` under a mode-`0700` directory. Its version-1
ownership envelope is the stable cross-version handoff boundary: an ACN retires itself when the
canonical registration no longer contains its exact opaque owner ID.

A client reuses the registration only when `/health` reports the same service, owner, version, and
PID in `Ready`. A matching `Starting` owner is joined rather than duplicated, and its installation
activity is reflected into the client lifecycle. Missing, invalid, unhealthy, or mismatched
registration is stale.

ACN binds its one final HTTP server and publishes that server's URL before preparing its private
ICN. The server serves canonical health and rejects RPC with `503` while starting. Once application
construction succeeds, the same router on the same server admits RPC and health reports `Ready`;
the URL, owner identity, PID, and registration never change. Health startup activity is
authoritative and may include byte progress only for actual artifact downloads.

## Starting one ACN

Before spawning, a client acquires a global spawn election and then checks registration again. This
prevents a client that waited for the election from acting on an earlier observation.

The candidate starts its final HTTP server and atomically becomes the canonical registered process
before acquiring active machine ownership. A predecessor observes that registration change and
requests its own scoped shutdown. The candidate reports that it is waiting and acquires machine
ownership only after predecessor application disposal and ICN reaping release it. It then constructs
its application and private ICN. This permits overlapping process shells but never overlapping
active ACN applications or ICNs.

The election is released as soon as the candidate publishes its early endpoint; readiness waiting
does not serialize other clients behind the election. A follower waits without a deadline while an
exact live election owner exists and can then join the registered starting ACN.

Election and ownership records contain a PID and unique identity. Stale recovery requires proof that
the PID is dead and removal of the exact observed record; timeout alone cannot steal a live record
or remove its successor. A candidate that loses canonical registration while waiting retires itself
and cannot retain active ownership.

## Version policy

| Client relative to healthy ACN | Result |
| --- | --- |
| Same version | Reuse |
| Older | Reuse the newer ACN |
| Newer | Start the expected ACN, which becomes canonical and causes self-retirement |
| Development build vs published release of the same base | Start development, which becomes canonical and causes the published ACN to self-retire |
| Published release vs development build of the same base | Reuse development |
| Same SemVer precedence with different build identities | Naturally order the build identities |
| Arbitrary non-SemVer identities | Naturally order the complete identities |

Release comparison is applied only after the observed health contract decodes and matches the
registered identity. Any health-incompatible incumbent is replaced by the expected ACN. Magnitude
development identities have the form
`<version>+dev.<commit>.<timestamp>` and are naturally ordered, including numeric timestamp ordering.
A development build outranks a published release with the same SemVer base, ensuring local
development replaces installed code rather than silently reusing it. A release with a newer SemVer
base still outranks an older development build. Every distinct identity has a deterministic order,
so version comparison cannot produce an unorderable conflict.

The client never asks an incumbent to shut down and never signals it as part of upgrade. The
incumbent's ownership watchdog is responsible for invoking its own shutdown semantics. This avoids
an upgrade command whose authentication and meaning would have to remain compatible across
arbitrary versions.

Candidate startup has a ten-second deadline only to publish registration and startup health. After
publication there is no global readiness or ownership-wait timeout. Exact candidate exit wins
immediately over registration polling. Platform process-spawn services continuously drain stdout
and stderr into a bounded diagnostic tail. Exact exit includes that diagnostic, while publication
timeout terminates and reaps the candidate before one final compatible-owner check. A live
predecessor is never considered stale because of elapsed time.

## Client startup presentation

The CLI calls `AcnStartup.prepare` before creating its renderer. A ready owner enters the ordinary
TUI without a bootstrap frame. Otherwise prepare starts the shared ensure attempt and waits only
for its first non-Checking lifecycle state, making the root bootstrap surface the first frame.
The bootstrap subtitle follows the authoritative startup phase: discovery looks for Magnitude,
ownership handoff waits for the previous Magnitude process, ACN launch starts Magnitude, and local
inference resolution and launch prepare and start local inference respectively. Clients do not
infer these phases from elapsed time or process observations.

`Starting Magnitude` covers cached process startup. `Installing Magnitude` has one bar and only
three subtitles: `Downloading daemon`, `Downloading inference engine`, and `Starting Magnitude`.
The first 90% is bundle-size-weighted download progress; the final 10% is asymptotic startup
progress. A provisional cross-platform accelerator reservation is marked non-authoritative and is
replaced by the exact selected CPU/CUDA/Vulkan plan before accelerator acquisition. Failure replaces
the surface with the causal safe error and Retry/Quit actions. Ctrl-C remains available throughout.
Individual commands and screens do not own startup hooks, retry loops, or copies of lifecycle state.

When a selected local backend requires startup preparation, ICN reports that operation before its
readiness record and ACN mirrors it as a structured starting phase. CUDA preparation includes its
detected hardware label and renders as `Preparing CUDA backend for <hardware>`. This remains under
`Starting Magnitude`; it is not an installation phase and carries no fabricated percentage.

## Recovery and compatibility

Transport loss invalidates the failed endpoint and may start ACN. A subscription receiving
`terminated` waits for another registered ACN without starting one. Protocol errors are surfaced
without invalidation, spawn, or downgrade. Recovery from retirement does not immediately reconnect
to the draining URL.

Forward reuse requires newer ACNs to preserve released request and response meanings plus the
registration, health, and subscription fields used by older clients. An incompatible health
contract causes the client to start its expected ACN; self-retirement still depends only on the
stable registration owner ID.

Downloaded ACN binaries use manifest-declared digest paths under their version and host, with a
small validated pointer selecting the current digest. ACN receives the same data root used for
registration, election, ownership, storage, and ICN storage.
