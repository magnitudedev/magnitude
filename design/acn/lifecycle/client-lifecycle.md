---
applies_to:
  - packages/sdk/src/client.ts
  - packages/sdk/src/service-starter.ts
  - packages/sdk/src/connection-errors.ts
  - packages/sdk/src/jit-rpc/**
  - packages/client-common/src/connection/**
  - packages/client-common/src/state/service-startup.ts
  - packages/client-common/src/state/service-recovery.ts
  - cli/src/server/acn-connection.ts
  - desktop/src/*.ts
  - web/src/platform/**
---

# SDK connection and first-party presentation

The SDK owns one scoped application connection. Construction and observation are passive. The first
operation, or explicit `connection.connect`, performs admission. Its state is
`Idle | Connecting(reason, activity?) | Ready(service) | Failed(error) | Closed`.

## Admission and startup

Admission probes public health at the configured origin, normally loopback port 10100. A Ready
response must supply the exact supported RPC version and an instance identity before RPC is
dispatched. HTTP success, a process, or completion of a starter is not sufficient. A daemon that
predates the RPC version answers the older health shape and is read as speaking version 0. Protocol
mismatch is a typed error; the SDK itself never negotiates, upgrades, downgrades, or replaces
software.

Every probe observes a usable service, ready or still starting, or an unusable one: absent,
undecodable, or the wrong protocol. An unusable service may invoke an injected
`MagnitudeServiceStarter`. The SDK invokes it at most once per admission occurrence, observes
progress, then verifies public readiness; after it has run, only a transient absence is tolerated
and any other answer is final. The host decides what the starter does: the CLI's replaces an older
daemon, the plugin's asks the installed CLI. An already-starting service is observed without
invoking another starter. Connect-only clients omit that capability and fail with the typed error.
The default absolute admission deadline is ten minutes, including starter execution and health
waiting. Each health request has a two-second bound.

The CLI starter requires only the abstract Effect Platform CommandExecutor. It runs argv
`magnitude service start`, drains human output, retains bounded stderr for errors, and owns the
command's cancellation. It neither uses a shell string nor parses a model-control JSON protocol.
Missing executable, failed command, unavailable service, malformed health, and protocol mismatch
remain distinguishable failures.

Privileged first-party hosts can supply a direct starter backed by private daemon-management.
That package alone owns SQLite coordination, exact-process supervision, binary acquisition, and
OS service administration. The SDK receives progress or failure, never owner rows or launch
targets. Desktop and web host bridges carry the same startup-only stream; application RPC remains
on the existing daemon endpoint.

## Concurrency and recovery

Concurrent bootstrap, operation, retry, and recovery calls join one shared admission outcome.
Cancelling a waiter does not cancel that shared attempt. Closing the SDK cancels it and
terminalizes all waiters. Admission publication and close share one serialized boundary.

Every RPC carries the selected instance ID; a successor at the same address rejects stale
dispatch. Confirmed transport loss re-enters admission. Domain errors and caller cancellation
do not. Finite operations follow their declaration's replay policy: replay-safe reads may retry;
ambiguous at-most-once mutations report an unknown outcome rather than duplicate side effects.
Finite declarations must apply `replaySafe` or `atMostOnce` before entering the RPC tree. The type
system enforces this, and group construction rejects omissions at runtime. Stream recovery follows
the stream protocol rather than a finite replay annotation. Error adaptation occurs once while
mapping the RPC dispatch function onto the SDK namespace, for both unary and streaming calls.
Subscriptions reopen and reread authority. Repeated attempts without meaningful progress have a
finite bound; a healthy quiet subscription is kept alive by transport frames.

Recovery retains the SDK instance, query cache, UI state, and registry. Client-common owns all
query invalidation and presentation policy. Reconnection invalidates first-party reads because
change notifications may have been missed.

## First-party presentation

Client-common projects SDK connection state into startup `Checking | Starting | Installing |
Ready | Failed`. Warm startup can move directly from Checking to Ready without painting progress.
Installation observations preserve measured download weighting and estimated startup progress.
A failed retry returns to Checking; initial Ready is terminal.

After initial readiness, recovery has a separate occurrence and fresh lifecycle. Existing UI
notification areas consume `Inactive | Recovering(lifecycle) | Recovered`; they do not remount
the application or implement another admission/retry engine.

`FirstPartyConnection` exposes `ServiceStartup`, `ServiceLifecycle`, and `ServiceRecovery`.
One scoped observer folds SDK changes into presentation history. The pure reducer consumes
`Connecting.reason`; it does not infer recovery from independent mutable connection flags. Failed
recovery and its retries share one notice occurrence, and a later recovery gets a new occurrence.
Download weighting and clock-driven estimates are pure rendering of that history. Observation
timers are switched with the current presentation and need no per-recovery fiber bookkeeping.
The lifecycle module declares whether a model's rendering depends on time; connection composition
does not inspect installation phases to decide when to tick.

CLI update discovery and the update-before-download gate remain before bootstrap. Renderer
construction remains after readiness. Logging, appearance probing, and shutdown ordering retain
their existing ownership.

Electron progress and errors are schema-encoded before structured cloning and decoded afterward.
Live Effect Option values and class instances never cross the bridge. Cancellation closes the
host observation, not an admitted daemon.

## Close

Close is one-way and idempotent. It stops client-owned transport, admission, and observers only.
It never starts, stops, replaces, or retains ACN, ICN, or models. If close wins the admission race,
no Ready endpoint can be published afterward.

## Conformance

- Passive construction and observation issue no requests or commands.
- Only matching Ready health admits operations.
- Concurrent callers share one startup; waiter cancellation preserves other callers.
- Scope close cancels startup and prevents subsequent admission.
- Replacement is fenced by instance ID and rechecks the protocol before redispatch.
- Startup failure and subsequent recovery are retryable on the same SDK instance.
- Client-common state and query ownership survive daemon recovery.
