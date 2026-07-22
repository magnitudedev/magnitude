---
applies_to:
  - packages/sdk/src/jit-rpc/**
  - packages/sdk/src/acn-jit/**
  - packages/sdk/src/daemon-spawner.ts
  - packages/sdk/src/local-spawner.ts
  - packages/sdk/src/remote-spawner.ts
  - packages/sdk/src/binary.ts
  - packages/protocol/src/acn-registry.ts
  - cli/src/platform/**
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
---

# Shared ACN startup and upgrades

All clients using the same data root share one ACN. That ACN owns one private ICN.

## Connecting from a client

Each client process creates one `AcnJitRuntime`. All RPC consumers share its protocol layer,
coordinator, and cached endpoint. Runtime construction ensures ACN once; concurrent recovery also
shares one discovery or spawn attempt.

Cached endpoints are generation-specific. A failed request can invalidate only the endpoint it used,
not a newer endpoint discovered by another request. Domain errors and caller cancellation do not
invalidate ACN.

## Finding the current ACN

The data root contains one canonical registration with ACN identity, version, URL, PID, and a private
shutdown token. Registration is atomically published as mode `0600` under a mode-`0700` directory.

A client reuses the registration only when `/health` reports the same service, owner, version, and
PID. Missing, invalid, unhealthy, or mismatched registration is stale.

## Starting one ACN

Before spawning, a client acquires a global spawn election and then checks registration again. This
prevents a client that waited for the election from acting on an earlier observation.

The ACN process separately acquires machine ownership before starting HTTP or ICN. This prevents
duplicate ICNs even after direct launch, client crash, or election failure. Ownership remains held
until ACN finalization and ICN reaping finish.

Election and ownership records contain a PID and unique identity. Stale recovery requires proof that
the PID is dead and removal of the exact observed record; timeout alone cannot steal a live record
or remove its successor.

## Version policy

| Client relative to healthy ACN | Result |
| --- | --- |
| Same version | Reuse |
| Older | Reuse the newer ACN |
| Newer | Replace through authenticated shutdown |
| Same SemVer precedence with different build identities | Naturally order the build identities |
| Arbitrary non-SemVer identities | Naturally order the complete identities |

The same comparison is used during discovery, after election, before shutdown, and while waiting for
a spawned candidate. Magnitude development identities have the form
`<version>+dev.<commit>.<timestamp>` and are naturally ordered, including numeric timestamp ordering.
A published release outranks a development build with the same SemVer base. Every distinct identity
has a deterministic order, so version comparison cannot produce an unorderable conflict.

A replacement client revalidates the incumbent and requests shutdown with its registered token.
Shutdown immediately cancels resident work. If the authenticated owner does not exit within a short
cooperative window, replacement escalates through `SIGTERM` and `SIGKILL` with bounded waits before
starting its candidate. Signals are never sent before authenticated shutdown acceptance. The
candidate must acquire machine ownership. If a newer candidate won meanwhile, the client uses that
winner instead.

The old ACN has a hard five-second retirement budget measured from authenticated shutdown
acceptance:

1. three seconds for cooperative cancellation, persistence flush, child cleanup, and normal exit;
2. one additional second after `SIGTERM`;
3. `SIGKILL` at four seconds if the process is still alive; and
4. one final second to confirm that the process was reaped.

Failure to reap the exact incumbent PID within that fifth second fails takeover. Candidate startup
has a separate ten-second registration-and-health deadline and does not extend the old ACN's
retirement budget.

## Recovery and compatibility

Transport loss invalidates the failed endpoint and may start ACN. A subscription receiving
`terminated` waits for another registered ACN without starting one. Protocol errors are surfaced
without invalidation, spawn, or downgrade. Recovery from retirement does not immediately reconnect
to the draining URL.

Forward reuse requires newer ACNs to preserve released request and response meanings plus the
registration, health, and subscription fields used by older clients. Breaking wire changes require
explicit compatibility negotiation.

Downloaded ACN binaries use immutable version/platform paths. ACN receives the same data root used
for registration, election, ownership, storage, and ICN storage.
