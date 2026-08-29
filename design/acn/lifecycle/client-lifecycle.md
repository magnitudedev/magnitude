---
applies_to:
  - packages/sdk/src/acn-jit/acn-recovering-client.ts
  - packages/sdk/src/acn-jit/acn-instance-manager.ts
  - packages/sdk/src/acn-jit/local-acn-instance-manager.ts
  - packages/sdk/src/acn-jit/acn-ensurance-coordinator.ts
  - packages/sdk/src/acn-jit/remote-acn-instance-manager.ts
  - packages/sdk/src/jit-rpc/**
  - packages/client-common/src/state/acn-recovery.ts
  - packages/client-common/src/state/acn-startup.ts
  - cli/src/features/app-shell/**
  - cli/src/platform/**
  - desktop/src/*.ts
  - web/src/platform/**
  - web/scripts/dev-server.ts
---

# ACN client lifecycle

Each interactive client owns one `AcnConnection`. The connection owns the client's effective ACN
identity, exact selected `AcnInstance<AcnReady>`, single-flight selection, recovering transport,
startup lifecycle observation, recovery occurrences, and one-way close. It does not interpret
coordination state, probe health, choose replacement, or manage processes; those belong to
`AcnInstanceManager`. Host adapters such as the terminal and browser platforms contain only
environment operations; they do not own service acquisition, transport, startup, or recovery.

These are presentation states, not another ACN service lifecycle.

## Presentation state machine

```text
initial
`-- [Checking]
    +-- startup observation ----------------------> [Starting]
    +-- installation observation -----------------> [Installing]
    +-- exact ready selection ---------------------> [Ready]
    `-- terminal selection failure ---------------> [Failed]

[Starting]
    +-- starting phase update ---------------------> [Starting]
    +-- installation begins -----------------------> [Installing]
    +-- exact ready selection ---------------------> [Ready]
    `-- terminal selection failure ---------------> [Failed]

[Installing]
    +-- download/progress update ------------------> [Installing]
    +-- non-download startup work -----------------> [Starting]
    +-- exact ready selection ---------------------> [Ready]
    `-- terminal selection failure ---------------> [Failed]

[Failed]
    +-- retry observes startup --------------------> [Starting]
    +-- retry observes installation --------------> [Installing]
    +-- retry selects an already-ready endpoint --> [Ready]
    `-- retry fails -------------------------------> [Failed]

[Ready]
    `-- terminal for startup; endpoint recovery uses a fresh occurrence lifecycle
```

Any non-ready state may reach `Ready` when exact endpoint selection succeeds.

## Bootstrap presentation phases

The startup lifecycle is a projection of authoritative observations available to the client. An
already-ready selection may move directly from `Checking` to `Ready`; selection does not synthesize
`PreparingAcn` merely to prove that it is active. This preserves the warm-start guarantee that the
CLI can check readiness without painting transient progress.

```text
Client bootstrap
|
+-- Checking                                      no deliberate wait; screen hidden
|
+-- Starting
|   |
|   +-- PreparingAcn                              "Preparing background server"
|   |   |
|   |   +-- read owner store                      normally immediate; SQLite contention <= 30s
|   |   +-- inspect exact process                 normally immediate; facility retry <= 30s
|   |   +-- classify owner/process group          one coordination pass
|   |   +-- probe owner health                    <= 2s per request
|   |   +-- wait between observations             1s polling interval
|   |   +-- tolerate unobservable live health     <= 30s
|   |   +-- supervise stale/obsolete daemon shutdown <= 2s shutdown request + 5s graceful
|   |   |                                           + 2s TERM + 2s KILL
|   |   +-- resolve daemon launch material         variable; ensurance remains <= 10m total
|   |   +-- spawn and inspect candidate            normally immediate
|   |   +-- await candidate owner admission        <= 30s
|   |   +-- retain replaced candidate exit proof   2s observation + 1s poll cycles; <= 10m total
|   |   `-- await first authoritative health       1s polling, <= 2s per request
|   |
|   +-- WaitingForOwner                          "Waiting for previous Magnitude process"
|   |   `-- daemon awaits ownership admission     <= 30s candidate admission bound
|   |
|   +-- ResolvingLocalInference                  "Preparing local inference"
|   |   +-- locate inference-server installation
|   |   +-- verify executable identity
|   |   `-- verify API, build, target, capabilities
|   |                                               all daemon startup work <= 5m total
|   |
|   +-- LaunchingLocalInference                  "Starting local inference"
|   |   +-- spawn inference server
|   |   +-- await and validate startup record
|   |   +-- validate loopback health identity
|   |   `-- commit readiness                       all daemon startup work <= 5m total
|   |
|   `-- PreparingBackend                         "Preparing <backend> backend for <hardware>"
|       +-- CPU
|       +-- Metal
|       +-- CUDA
|       `-- Vulkan                                 all daemon startup work <= 5m total
|
+-- Installing                                    "Installing Magnitude"
|   |
|   +-- DownloadingDaemon                         network-dependent; ensurance <= 10m total
|   +-- DownloadingInferenceEngine                network-dependent; daemon startup <= 5m total
|   `-- StartingMagnitude                         daemon startup <= 5m total
|
+-- FinalizingSelection                            no distinct display; previous phase remains
|   +-- revalidate exact owner and process         store errors terminal; process retry <= 30s
|   `-- publish exact RPC endpoint                 atomic with the open/closed check
|
+-- Ready                                          terminal successful presentation state
|
`-- Failed
    +-- InstallDaemon
    +-- LaunchDaemon
    +-- PrepareLocalInference
    `-- Connect

Absolute selection/ensurance deadline: 10m
```

The five-minute daemon-startup ceiling is shared by resolving, installing, launching, and backend
preparation; it is not a fresh five-minute allowance for each displayed phase. Likewise, the
ten-minute selection deadline bounds the complete manager occurrence rather than resetting for
each coordination substate. Network transfers have no independent fixed duration beyond those
enclosing absolute deadlines.

## Association and selection

```text
AcnAssociation
  identity    monotonic minimum ACN identity
  selected    optional exact AcnInstance<AcnReady>

ActiveSelection
  one shared deferred outcome
```

The association starts at the bundled SDK identity. Only successful ready selection adopts a newer
identity. The instance manager compares the client's target only with exact live owner health: it
adopts equal or newer revisions and replaces lower revisions. A historical or dead revision has no
authority, and losing the selected endpoint never regresses the client's effective identity.

Selection is a true single-flight operation. Bootstrap, retry, and application calls share
one scoped owner and one exact outcome while selection is active; a semaphore that merely queues
new operations is insufficient. The owner calls `AcnInstanceManager.ensure`, projects progress into
client presentation, and atomically publishes only terminal `AcnInstance<AcnReady>` values. Every
typed manager terminal failure is projected to `Failed`; explicit retry starts one new ensure
occurrence with a fresh absolute deadline.

Runtime construction explicitly starts initial selection. Selected-instance publication and the
open/closed check occur in one admission critical section.

## Recovery

Every RPC carries both URL and exact ACN instance ID. ACN dispatch rejects another occurrence.
Transport failure clears only the matching failed selection, joins or starts the same selection
single-flight, then retries the exact request according to its transport contract. Domain failure
and caller cancellation do not trigger recovery.

Once initial selection has succeeded, every replacement selection owns a fresh lifecycle and a
monotonic recovery occurrence. The runtime exposes `Inactive | Recovering(lifecycle) | Recovered`
separately from startup. Clients may project that occurrence into their existing notification area;
they must not move startup presentation backward, remount application UI, or duplicate selection
and retry logic.

A successful selection is a point-in-time fact. Retirement may begin after selection; exact request
addressing prevents misrouting and recovery handles that unavoidable race. Desktop and web preserve
the same typed ensure stream and cancellation semantics as local execution.

Electron's isolated renderer boundary is a real serialization boundary even though it exposes a
JavaScript callback facade. The preload encodes each ensure event with its canonical Effect Schema
before `contextBridge` clones it, and the renderer decodes it before handing it to the SDK runtime.
Effect data types such as `Option` must never be passed across that boundary as live objects because
structured cloning removes the symbols and prototypes required by their decoded representation.

Each concrete `RpcClient` owns its own single-consumer protocol receiver.

## Close

Close is one-way and idempotent. It marks the runtime closed under selection admission, closes the
selection scope, and awaits interruption. It performs no ACN lifecycle RPC. Close and scope
finalization never ensure, discover, replace, launch, or stop an ACN.

Selection publication checks `open` under the same admission boundary as close. If close wins, no
selection is published. Each host invokes runtime close as its
leading teardown action, explicitly or through its registered finalizer, before destroying runtime
dependencies.

## Guarantees

- Only `AcnInstance<AcnReady>` enters endpoint selection.
- Identity never regresses during one client lifetime.
- Initial selection, retry, and application recovery share one selection outcome.
- Transitional assignment and temporary health failure cannot independently become startup failure.
- Exact addressing prevents a stale selection from reaching a successor.
- Every Electron ensure event is schema-encoded before and decoded after structured cloning.
- Close cannot publish an endpoint or invoke ensurance after closing begins.
- Intentional replacement is not reported as a crash; process output remains diagnostic.
