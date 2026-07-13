# ACN Lifecycle

The ACN is the local daemon that hosts Magnitude's RPC server, session runtime, display streams, file watching, image description, and related long-lived services. Clients never manage the ACN's lifecycle explicitly. They fire operations; the SDK guarantees that an ACN exists, in some shape or form, by the time the operation needs one.

## The Operation Contract

This is the core contract between a client and the ACN. Everything else in this document supports it.

**The client opportunistically assumes the ACN might be alive.** It never verifies liveness before an operation. It dispatches over whatever connection it has cached; the cached connection is purely an optimization and must be invisible to semantics.

**Unreachability is the death signal, and it must surface fast.** A failure to reach the ACN — connection refused, connection reset, broken stream — means the ACN is dead. Because the transport is machine-local, these failures are effectively instant, and the client must not add waiting on top of them: no transport-level retries, no backoff loops against the same daemon. Operation *duration* is never a death signal — an operation may legitimately take minutes.

**On transport failure: recover and re-issue, exactly once.** The client immediately re-resolves a daemon (discover a healthy one, else spawn one) and re-issues the same operation against it. If the re-issued operation fails at the transport level again, the daemon is crash-looping and the failure is surfaced as fatal. This recovery is native to the SDK and enforced at a single choke point derived from the RPC group — every operation, including ones added in the future, recovers automatically. No user operation is ever dropped because the daemon happened to be dead.

**The ACN is free to terminate itself at any time, for any reason, without telling anyone.** Idle shutdown, self-displacement, a crash, a kill -9 — the client contract does not distinguish them. Graceful cleanup (deregistering, disposing sessions) is a courtesy that reduces recovery latency, not part of the contract. The obligation that replaces graceful shutdown is durability: anything the ACN has acknowledged must be persisted well enough that a successor daemon can serve the session from disk.

**Duplicate delivery across a daemon death is an accepted risk.** If the ACN processes an operation and dies before responding, the re-issued operation is delivered twice. The window is tiny for a local daemon. Idempotency keys can close it later if it ever matters.

## Error Taxonomy

SDK operations fail in exactly two ways:

- **Domain errors** — `SessionNotFound`, `InvalidSessionPath`, permission failures, and the like. These are real answers from a live daemon and pass through untouched. They are never retried and never trigger recovery.
- **Fatal infrastructure errors** — binary missing, binary version mismatch, spawn failure, or a daemon that dies again immediately after being respawned. These mean the machine cannot host an ACN right now and no amount of retrying fixes them. They are the only failures a consumer should treat as "the daemon is unavailable."

Transient transport errors do not exist in the public API. The only thing a caller can observe from a daemon death is a latency blip on the operation that hit it.

## Spawners

Resolution and spawning sit behind a two-method spawner abstraction — the single environment-specific point in the stack:

- `discover()` — find a healthy same-version daemon, or report none.
- `spawn(command?)` — converge on a healthy daemon and return its URL.

`spawn` means *converge*, not "always start a process": it serializes concurrent spawners (the local spawner uses a short-lived version lock), re-checks for a daemon that another client won the race to start, and only then starts one. Callers of the spawner never think about races.

Capability levels, all satisfying the same operation contract:

- **Local spawner** (CLI, Node/Bun SDK): reads the version registry, health-probes, spawns a detached process. Owns all filesystem and process mechanics described below.
- **Remote spawner** (browser): delegates `discover`/`spawn` over HTTP to a machine-local proxy that has the real spawn capability. The recovery engine above it is identical.
- **Discover-only / none**: a client that cannot spawn still recovers by re-discovering (someone else may have respawned the daemon); a client with a fixed URL and no spawner surfaces fatal unavailability immediately.

The recovery engine is written against the spawner interface and is shared, unchanged, across all environments.

## Version Model

ACN ownership is scoped by Magnitude version. A client resolves the daemon for its own target version and ignores daemons registered for other versions.

The invariant is:

- one healthy ACN per Magnitude version;
- multiple Magnitude versions may have ACNs running at the same time;
- clients connect only to the ACN for their target version;
- same-version ACNs compete for one version-local registration, while different-version ACNs do not displace each other.

The version namespace is represented under the user's Magnitude data directory. Each version has its own coordination directory containing a registry and a short-lived spawn lock.

## Local Spawner Mechanics

These mechanics are the local spawner's implementation of `discover`/`spawn`. They are not part of the operation contract.

### Registry

Each version registry records the active ACN identity for that version: daemon id, Magnitude version, protocol information, URL, process id, and timestamp. Registries are written atomically (write temporary file, rename into place) and are private to the user. On graceful shutdown, a daemon removes its registry only if the registry still points to its own daemon id.

An empty, missing, invalid, or unhealthy registry is unusable. `discover` reports none; `spawn` removes it and starts a replacement.

### Discover

1. Read the target version's registry.
2. Probe the registered daemon's health endpoint (short, local-scale timeout).
3. Report the daemon only if it is healthy, reports the target version, and its scheduler heartbeat is fresh — a wedged daemon is a dead daemon.

### Spawn

1. Acquire the target version's spawn lock.
2. Re-run discover inside the lock — another client may have won the race; if so, return that daemon's URL.
3. Remove the stale registry, start the ACN process detached with registration enabled.
4. Wait for it to write its registration and become healthy; if the process exits first or never becomes healthy within the spawn timeout, fail with a spawn error (fatal).
5. Release the lock. The lock coordinates spawning only; it is never a lease for serving traffic.

The spawner resolves the ACN binary before spawning: an explicit binary path, an already cached binary for the expected version, or a download of the expected version for the current platform. An explicit binary is verified by asking it to print its version.

## Multiple Versions

Different versions intentionally coexist. A newer client must not kill, overwrite, or reuse an older version's ACN, and vice versa. An upgrade can temporarily leave more than one ACN running; each serves its own version until normal lifecycle rules (idle timeout, self-termination, same-version displacement) retire it.

Cross-version session migration is not part of the lifecycle contract. A session belongs to the daemon/version serving it unless another system explicitly migrates it.

## Same-Version Displacement

Same-version daemons do not intentionally coexist. Every registered daemon has a unique owner id. While registered, a daemon periodically re-reads its own version registry; if it points at a different owner id, the daemon treats itself as displaced and shuts down. Displacement is one of the many ways an ACN may die — clients recover from it like any other death.

## Termination And Idleness

The ACN does not track connected clients as a correctness or ownership primitive, and it owes clients nothing at shutdown. When it chooses to exit is internal policy, informed by ACN-owned activity:

- active foreground agent work;
- active display stream subscribers;
- active long-running RPCs or subscriptions;
- recent command activity.

RPC activity is tracked by Effect RPC middleware on non-health operations. Display subscriber activity is tracked by the display stream service itself. Health checks and connection negotiation do not keep the daemon alive.

During graceful shutdown the daemon first removes its registration (only if it still owns it — a draining daemon must not be discoverable), then disposes live runtime sessions — but clients must tolerate the daemon skipping all of this. Work that was in flight and not yet persisted when the daemon died is lost; the successor daemon rehydrates the session from disk and the display stream resync shows the truth. That is the durability contract's flip side and it is acceptable.

## Streams And Liveness

Display streams and file watches are long-lived operations and follow the operation contract like everything else, with one addition: the client must be able to distinguish "daemon dead" from "no events."

- **Streams are resident: there is no legitimate server-initiated end.** A display view or file watch never completes from the ACN's side. A clean protocol exit or a server-side interrupt on a stream — which is exactly what a gracefully shutting-down or idle-timed-out daemon sends as it drains — is the daemon *relinquishing* the stream, and the client recovers from it exactly as from a transport death. A consumer's stream must never silently complete because the daemon went away; that would freeze every attached display with no error and no recovery.
- **Transport EOF without an explicit exit is a transport failure**, not a completed stream — same recovery.
- **Every long-lived server stream emits a heartbeat at a fixed cadence.** A client that sees no events (heartbeat or otherwise) for a small multiple of the cadence treats the stream as dead. Heartbeats are filtered out by the SDK and never reach consumers.
- On death or relinquishment, the client recovers exactly as for a unary operation: re-resolve, re-open. A display stream carries its view shape in the open request, so a stream is self-describing: neither the ACN nor the agent remembers per-view configuration for views that are not open, and a reconnect re-establishes the view entirely from the retried request.

The only things that terminate a stream toward the consumer are domain errors (for example `SessionNotFound`) and defects; they never trigger reconnection. Client-initiated close (unsubscribe) is fiber interruption on the client side and does not involve a server exit at all.

## Operational Expectations

Healthy behavior looks like this:

- killing an ACN mid-session is indistinguishable, to the user, from never having killed it — the next operation (or the stream watchdog) respawns it and the session rehydrates from disk;
- a client for version `V` always reads and writes only version `V` coordination state;
- starting version `W` does not displace version `V`;
- stale same-version registrations are replaced;
- concurrent same-version clients converge on one daemon;
- graceful shutdown does not erase another daemon's ownership;
- SDK consumers never write recovery logic, never see transport errors, and only handle domain errors and fatal unavailability.
