---
applies_to:
  - packages/sdk/src/acn-jit/acn-recovering-client.ts
  - packages/sdk/src/acn-jit/acn-instance-manager.ts
  - packages/sdk/src/acn-jit/local-acn-instance-manager.ts
  - packages/sdk/src/acn-jit/acn-ensurance-coordinator.ts
  - packages/sdk/src/acn-jit/remote-acn-instance-manager.ts
  - packages/sdk/src/jit-rpc/**
  - packages/client-common/src/state/acn-lifecycle.ts
  - cli/src/features/app-shell/**
  - cli/src/platform/**
  - desktop/src/*.ts
  - web/src/platform/**
  - web/scripts/dev-server.ts
---

# ACN client lifecycle

Each interactive client owns one `AcnJitRuntime`. The runtime owns the client's effective ACN
identity, exact selected `AcnInstance<AcnReady>`, single-flight selection, recovering transport,
`ClientId`, `ClientLease`, bootstrap presentation, and one-way close. It does not interpret
coordination state, probe health, choose replacement, or manage processes; those belong to
`AcnInstanceManager`.

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
    `-- terminal; endpoint recovery does not move presentation backward
```

Any non-ready state may reach `Ready` when exact endpoint selection and initial lease establishment
succeed.

## Bootstrap presentation phases

The bootstrap presentation is a projection of the latest authoritative observation available to
the client. A parent phase remains visible while work below it has not yet published a more specific
observation. In particular, `PreparingAcn` means that endpoint selection is active but the instance
manager or daemon has not yet published a finer startup observation; it does not mean that the
manager is only reading the owner store.

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
|   +-- connect exact RPC endpoint                 transport-bounded
|   `-- establish initial client lease             <= 5s per attempt; 250ms retry delay
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

Selection is a true single-flight operation. Bootstrap, retry, lease, and application demand share
one scoped owner and one exact outcome while selection is active; a semaphore that merely queues
new operations is insufficient. The owner calls `AcnInstanceManager.ensure`, projects progress into
client presentation, and atomically publishes only terminal `AcnInstance<AcnReady>` values. Every
typed manager terminal failure is projected to `Failed`; explicit retry starts one new ensure
occurrence with a fresh absolute deadline.

Runtime construction explicitly starts initial selection. It also constructs one inert, scoped
lease owner whose renewal fiber is gated by a deferred. Immediate lease establishment, selected
instance publication, and the open/closed check occur in one admission critical section. The
heartbeat starts only after that establishment succeeds.

## Recovery

Every RPC carries both URL and exact ACN instance ID. ACN dispatch rejects another occurrence.
Transport failure clears only the matching failed selection, joins or starts the same selection
single-flight, then retries the exact request according to its transport contract. Domain failure
and caller cancellation do not trigger recovery.

A successful selection is a point-in-time fact. Retirement may begin after selection; exact request
addressing prevents misrouting and recovery handles that unavoidable race. Desktop and web preserve
the same typed ensure stream and cancellation semantics as local execution.

Electron's isolated renderer boundary is a real serialization boundary even though it exposes a
JavaScript callback facade. The preload encodes each ensure event with its canonical Effect Schema
before `contextBridge` clones it, and the renderer decodes it before handing it to the SDK runtime.
Effect data types such as `Option` must never be passed across that boundary as live objects because
structured cloning removes the symbols and prototypes required by their decoded representation.

Each concrete `RpcClient` owns its own single-consumer protocol receiver. The private lease client
and application clients share semantic selection/recovery authority, never a protocol receiver.

## Close

Close is one-way and idempotent. It marks the runtime closed under selection admission, closes the
selection scope and awaits interruption, stops heartbeat renewal, then freezes the selected exact
endpoint. Bounded model observation and lease release use a non-recovering protocol bound only to
that endpoint. Those bounded shutdown RPCs run in a fresh operation scope owned by close; they never
acquire resources into the runtime's owning scope, which may already be finalizing. Close and scope
finalization never ensure, discover, replace, or launch an ACN.

Selection publication and lease establishment check `open` under the same admission boundary as
close. If establishment wins, close observes that exact selection and releases its lease; if close
wins, no lease is established or selection published. Each host invokes runtime close as its
leading teardown action, explicitly or through its registered finalizer, before destroying runtime
dependencies.
Browser back/forward-cache suspension is not close; lease expiry remains authoritative if renewal
cannot run while suspended.

## Guarantees

- Only `AcnInstance<AcnReady>` enters endpoint selection.
- Identity never regresses during one client lifetime.
- Initial selection, retry, lease recovery, and application recovery share one selection outcome.
- Client presence does not implicitly own bootstrap policy.
- Transitional assignment and temporary health failure cannot independently become startup failure.
- Exact addressing prevents a stale selection from reaching a successor.
- Every Electron ensure event is schema-encoded before and decoded after structured cloning.
- Close cannot publish an endpoint, create a lease, or invoke ensurance after closing begins.
- Intentional replacement is not reported as a crash; process output remains diagnostic.
