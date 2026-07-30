---
applies_to:
  - packages/icn-protocol/**
  - packages/icn/**
  - packages/openapi-effect/**
  - inference/crates/icn-api/**
  - inference/crates/icn-contracts/**
  - inference/crates/icn-server/**
  - packages/acn/src/server.ts
  - packages/acn/src/icn/**
  - packages/acn/src/serve.ts
  - packages/acn/src/binary.ts
  - packages/acn/src/daemon-lifecycle.ts
  - packages/acn/src/model-*.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/local-model-packages.ts
  - packages/acn/src/local-model-evaluations.ts
  - packages/acn/src/local-inference-hardware.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - inference/scripts/**
  - inference/package.json
  - package.json
  - scripts/dev.ts
  - inference/catalog/**
  - .github/workflows/release.yml
  - .github/workflows/release-build.yml
---

# ICN process lifecycle and Bun boundary

ICN is ACN's sole local-inference runtime. Every running ACN owns exactly one private ICN child
process for its entire ready lifetime. `@magnitudedev/icn-protocol` is the generated wire boundary
for both bootstrap records and the HTTP API. `@magnitudedev/icn` is the authored Effect-native
client integration and lifecycle manager for the child. Lifecycle code resolves and verifies the
native binary, owns the process scope, instantiates one `IcnClient` from the generated client
factory, observes termination, and performs bounded shutdown. ACN consumes that scoped client
directly; a second semantic facade is forbidden unless it owns a new invariant rather than merely
renaming generated operations.

ICN is not a separately discovered daemon. Clients never connect to it directly, and ACN does not
adopt an ICN started by another process. A model is not a public process resource: the one ICN
service starts without a loaded model and privately creates or destroys a disposable inference
worker behind model-centric load, replace, and unload operations.

The service process may own bounded, private native children. Planner workers are short-lived and
metadata-only. The inference worker owns the one resident model topology and lives only for that
residency generation. Both use the same verified executable, communicate only with their parent
over private standard I/O, expose no listener or public lifecycle, and terminate with their request,
residency, or parent. ACN still owns and observes exactly one ICN service child.

The hardware, fitting, and inventory meanings exposed through this boundary are defined by
[hardware fitting](./hardware-fitting.md) and [model management](./model-management.md). This
document defines ownership, process lifetime, transport, and the Bun-facing package contract.

## Ownership boundary

The ICN package owns:

- resolving a release-matched or explicitly configured ICN installation path;
- verifying its build and API compatibility before making it available;
- constructing the model-free server launch configuration;
- spawning exactly one child in an Effect scope;
- obtaining a race-free loopback address and proving that the endpoint belongs to that child;
- readiness, bounded diagnostics, structured child logging, exit observation, and shutdown;
- consuming the generated ICN API without an ICN-specific transport implementation;
- exposing one scoped generated client whose admitted streams preserve their response lifetime; and
- supplying the generated client with the connection established by the managed child lifecycle.

The package owns exact hardware, recommendable-catalog, installed-package, and download observation
plus the local provider adaptation because those capabilities compose the one generated ICN client.
It does not own recommendation policy, durable offerings, user selection, ACN RPC state, cloud
provider routing, or client presentation. It does not independently inspect hardware or GGUF files
in Bun; it obtains those facts through generated ICN operations. It contains no fallback
implementation of an ICN operation and no hand-written HTTP client, SSE parser, wire schema, or
endpoint-specific transport error mapper.

The native ICN process owns hardware discovery, model acquisition and inventory, artifact
inspection, model fitting, the pinned inference runtime, active-model state, and inference request
execution. `@magnitudedev/icn` acquires a release-manifest base plus an optional concrete accelerator
pack; ICN is not downloaded from a model repository or selected from a user-installed runtime.
The installation carries the release-bound catalog lock and hardware-independent planner-input
bundle as integrity-checked sidecars. Native startup validates their source, template, planner,
integrity, and exact-coverage identities before becoming ready. Ordinary startup and setup
therefore do not contact a catalog service, fetch model headers, or depend on a user cache.
Development hydration and release CI obtain the sidecars explicitly from immutable catalog
revisions; ordinary TypeScript and Cargo builds perform no catalog network access.

ACN owns the parent scope and application policy. It supplies ICN's storage roots and supported
binary identity and translates private ICN observations into product-owned inventory, hardware,
catalog, and slot mirrors. Raw ICN schemas and terminology remain private; ACN does not expose
source-backed ICN mirrors to clients.

## Generated API boundary

`@magnitudedev/openapi-effect` generates the complete callable Effect API, not merely schemas and
server declarations that require a package-specific wrapper. From one normalized protocol IR it
emits:

- decoded and encoded Effect Schema types;
- server-side `HttpApi` declarations for transports Effect Platform can represent;
- complete operation metadata;
- a typed client whose ordinary operations return `Effect` and whose streaming operations admit
  the response in `Effect` before returning accepted response metadata and an event `Stream`;
- generated client service/tag and construction APIs; and
- a manifest binding every emitted artifact and operation to the source protocol.

The ICN bootstrap protocol comprises four non-HTTP records: binary identity, backend eligibility,
installation declaration, and process readiness. These records are canonical serializable Rust
types whose OpenAPI components generate the Effect Schemas consumed by Bun lifecycle, development,
and release tooling. Producers construct the canonical Rust types and TypeScript consumers decode
or encode only through the generated schemas; independently authored wire shapes, compatibility
aliases, parallel public schema surfaces, or unchecked JSON casts are forbidden.

The generator's shared runtime owns request encoding, path/query/header serialization, HTTP
execution, response decoding, SSE and NDJSON framing, declared stream termination and reconnect
policy, response-body cleanup, cancellation, and the common transport/protocol error model. That
runtime is part of `@magnitudedev/openapi-effect`; none of it is recreated in `@magnitudedev/icn`.

The generated client preserves per-operation success and declared error types and groups methods by
the OpenAPI operation groups. It captures `HttpClient` and connection configuration when
constructed so methods require no platform services. Streaming admission failures remain in the
outer `Effect`; only failures after an accepted response inhabit the returned `Stream`. Invalid
local input, transport failure, a declared remote response, an undeclared or malformed response,
and incomplete stream termination remain distinct generated client failures with their actual
response metadata.

Every operation in the normalized IR is emitted into the callable client automatically. There is
no allowlist or hand-maintained facade coverage table. At minimum the ICN contract comprises health
and identity, hardware, live Hugging Face discovery, recommendable catalog, installed packages,
assessment, automatic fitting, downloads, exact configuration load, exact model-instance Stop, template
application, model properties, and streamed chat completion. Generator tests
prove that the manifest, descriptors, and callable client contain the same operation set.

The ICN protocol package checks in all generated schemas, operations, server declarations, client,
and manifest from one Rust OpenAPI document. The ICN client package's authored boundary exposes
`IcnProcess`, which proves one ready scoped child, and `IcnClient`, which requires that process and
constructs the generated callable API. The generated service tag and standalone live layer are not
public construction paths because they would permit a client without process ownership. No
hand-authored model-serving facade duplicates or renames generated operations. The
admitted stream owns its response body until that Stream terminates or is canceled.

ICN also exposes a private revisioned model-instance snapshot and coalescing invalidation watch.
The authored ICN package admits the watch before fetching its initial snapshot, refetches on every
newer invalidation, and converges from a fresh invalidation after reconnect. A transient snapshot
failure is retried without abandoning its invalidation, and a terminal watch failure re-admits the
watch before refreshing current state. An explicit exact-instance check preserves refresh failure
instead of authorizing from a cached snapshot. ACN is the only consumer; native instance types
are not a client-facing product mirror. Instance identity, allocation, release, and failure
semantics are defined by
[local model instance lifecycle](./model-instance-lifecycle.md), not by the hardware API.

Chat's `[DONE]` sentinel and download's successful EOF are OpenAPI extension semantics consumed by
the generator. They are not ICN-specific parser branches.

## Configuration

Launch configuration and model execution configuration are separate.

The launch configuration contains only process-lifetime facts: binary resolution policy, loopback
binding, model-store and cache roots, optional read-only import/source roots, startup and shutdown
deadlines, output bounds, authentication/instance identity, and compatible API/build identity.
It must be validated before spawning.

The model store and disposable cache are separate roots. In the managed product layout, authoritative
model artifacts live under `.magnitude/models` and every Magnitude-owned disposable cache namespace
lives under `.magnitude/cache`; cache implementations must not create private cache roots beneath
the model store. ICN's managed Hugging Face hub lives beneath the model store, and ICN does not
implicitly discover or adopt a host user's global Hugging Face cache.
External caches or directories participate only when they are supplied explicitly as read-only
import/source roots. ACN supplies no such roots for the product-managed ICN.

Per-request context length belongs to an explicit model serving configuration supplied to
assessment, fitting, and load. ACN persists that configuration inside a provider offering; ICN owns
its identity and ephemeral residency. Serving configuration is not an installation-manifest, cache,
or process-launch field. Native sequence capacity, physical context allocation, batching, GPU
placement, KV policy, projector, draft, and MTP selection are ICN-owned plan resolution. This
separation lets one ICN live for one ACN lifetime while models and configurations change
independently.

Runtime code receives configuration explicitly and uses Effect platform services for command
execution, filesystem/path work, HTTP, clock, randomness, logging, and scope. Core lifecycle code
does not reach directly into Bun globals, Node port-probing APIs, environment variables, or the
user's home directory. A Bun composition layer may translate process environment and packaged
paths into the typed configuration. The managed child disables implicit Hugging Face credentials;
native Hub access may use an explicitly supplied token but never a host login discovered from a
global token file.

ACN-owned ICN shutdown is bounded to one second: ICN receives `SIGTERM` and 500 milliseconds to
exit, then receives `SIGKILL` and has another 500 milliseconds to be reaped. Model loading,
inference, and model-resource release do not extend this deadline.

## Binary resolution and compatibility

Production releases publish a CPU-capable ICN base per host and distinct accelerator packs. The ICN
package shares release-manifest validation, bounded download, safe extraction, and digest-addressed
artifact installation with CLI and SDK acquisition. It alone owns native
eligibility probing, concrete backend resolution, base/pack composition, installation declaration,
and native validation.

Production has no requested or cached `auto` policy. Apple arm64 requires Metal. Other hosts prefer
compatible CUDA, then compatible Vulkan, and select CPU only after successful probes establish
that no supported accelerator is usable. Probe or operational failure fails ACN startup rather than
changing backend. Every supervised start probes again before deriving the concrete composition
identity, so installing a driver can change the next composition without manual cache repair.

An installation is immutable and identified by the release manifest, base, optional pack, concrete
backend, native build, and backend-module ABI. Its fixed layout contains executable,
runtime, backend modules, catalog sidecars, and a minimal declaration. Native validation proves the
running executable belongs to that installation, the selected backend directory contains only the
declared accelerator family, required devices register, and catalog sidecars are complete.

`bun dev` prepares the same fixed layout at
`inference/target/development/installation.json` before starting the client. `MAGNITUDE_ICN_PATH`
may instead name another `installation.json`; no separate executable or runtime path exists.
Development preparation may accept an explicit backend override, but that policy is not part of
production release coordinates.

Compatibility is established by a versioned ICN API protocol identity plus the release's expected
native build identity. It is not inferred merely because `/health` returned 200, and it need not
require unrelated package semvers to be textually equal. Development installations still enforce
the supported API identity and required capabilities.

## Lifecycle state machine

The lifecycle has the following states:

```text
Resolving -> Starting -> Ready -> Stopping -> Exited
     |          |                    |
     +----------+--------------------+-> StartupFailed / ShutdownFailed

Ready -- unexpected child exit --> DependencyFailed --> ACN termination
```

Only `Ready` publishes `IcnProcess`; only that capability permits construction of `IcnClient`.
The process service exposes immutable child identity and exit observation. Starting, stopping, and failed states are lifecycle
observations and errors, not partially usable clients.

### Startup

Startup is one scoped acquisition:

1. Validate launch configuration, resolve the executable, and verify its identity.
2. Create a fresh opaque instance ID and child-only authorization capability.
3. Spawn `magnitude-icn serve` model-free, bound to loopback port zero. ICN asks the operating
   system for the port; Bun must not probe a free port and release it before spawn.
4. Consume stdout and stderr through one supervised pipeline. Retain a bounded diagnostic tail and
   forward line-oriented output to structured logs without secrets.
5. Read ICN's machine-readable startup record containing the actual origin, instance ID, process
   identity, API identity, and native build identity. Arbitrary human log text is not a readiness
   protocol.
6. Probe readiness with bounded backoff while racing the child exit and overall startup deadline.
   Validate the instance ID and compatibility fields so an unrelated listener can never satisfy
   readiness.
7. Publish `IcnProcess`, construct `IcnClient` from it, and begin continuous exit supervision.

ICN's HTTP listener is created before it emits the startup record. Its readiness response is
successful only after storage, inventory recovery, native runtime registration, and API state are
usable; it does not perform an inventory-wide model inspection. Startup retry applies only to
transient connection/unready outcomes. Authentication failure, instance mismatch, incompatible
identity, malformed response, and child exit fail immediately.

Before this acquisition begins, ACN publishes its owner registration and a startup-only health
endpoint. That endpoint reports the resolver's authoritative base download, accelerator download,
installation, and launch activity; byte progress is present only while artifact bytes are being
accepted. It rejects application RPC until acquisition succeeds. ACN reports `Ready` and admits RPC
only after `IcnProcess` and the complete application layer exist. Concurrent consumers share the
same memoized layer and cannot create additional children.

### Ready lifetime and unexpected exit

The process handle, output fibers, generated client, and request scopes all descend from the same
ICN scope. Dropping an individual HTTP stream cancels and closes that response; it does not stop the
child. Closing the ICN scope cancels every in-flight ICN request before process termination.

There is no automatic in-process ICN restart. If the child exits unexpectedly, ACN has lost a
required authority and must fail closed: the supervisor records the exit status and bounded
diagnostic tail, fails new and in-flight ICN work with a typed dependency failure, and initiates ACN
termination. The external ACN recovery path may then start a fresh ACN/ICN pair. This avoids
preserving provider sessions, model-operation assumptions, or streams across an unannounced native
runtime replacement.

Child exit is considered expected only after scoped shutdown has entered `Stopping`. Exit code zero
before readiness or during `Ready` is still unexpected.

### Shutdown

ACN first closes root demand admission and finalizes ICN request producers and observers. Explicit
shutdown and scope finalization use the same idempotent child operation. ACN ownership remains held
until that operation finishes.

1. Mark ICN stopping so exit is no longer classified as dependency loss.
2. Send the platform's graceful termination signal once.
3. Wait a bounded grace period for ICN to stop accepting work, cancel operations, flush safe
   inventory state, close its listener, and exit.
4. If the deadline expires, send a forceful termination signal, wait a second bounded period, and
   reap the child.
5. Finalize process-output observers and publish the terminal lifecycle result.

The native server must handle both interrupt and termination signals with the same idempotent
graceful-shutdown path. Repeated shutdown requests do not send overlapping signal sequences.
Finalization is uninterruptible around signal delivery and child reaping, while both waits remain
bounded. Cleanup errors are logged and classified; they never leave an unobserved child handle.

Signals and recoverable ownership loss use ACN's common coordinator; they do not call `process.exit`
before cleanup. For abrupt death, ICN watches parent-PID liveness and EOF on a private inherited
stdin pipe. The EOF wait runs on a detached OS thread, not Tokio's blocking pool. These are crash
backstops, not child adoption or sharing.

## Model instance lifecycle

The singleton starts with no model instances. Catalog, installed-package, assessment, fitting,
download, and deletion remain available in that state. ICN's `ModelInstanceController` owns
physical instance admission, native workers, backends, exact-instance leases, lifecycle
publication, and terminal cleanup. ACN owns product slots and explicit load/Stop policy. A slot is
not an ICN resource.

Model load accepts one exact model serving configuration plus the branded model-instance identity
created by ACN before admission. The configuration contains the target
and per-request context length; ICN owns its stable identity and ACN passes it unchanged from the
selected provider offering. After proving the exact one-sequence baseline, load selects the largest
native sequence capacity from one through four whose full-context allocation fits stable and live
memory policy. That resolved capacity belongs to residency execution evidence and may differ across
cold loads of the same configuration.
Load does not accept a planner name, planner version, capacity-policy identifier, or native flags.
ICN reassesses the exact plan and streams typed progress through resolution, assessment,
unload/replacement, loading, verification, and ready or failed termination. Loading percentage
begins only after the exact native plan is prepared and prior residency is released. ICN estimates
total progress from the prepared plan's semantic phase sequence and phase-duration estimates,
keeps it monotonic, and caps it below completion; only Ready means complete. Loading, progress,
Ready, Stopping, Stopped, and Failed are published in the revisioned
`ModelInstancesSnapshot`. ICN admits one exact instance identity idempotently; a later load of the
same configuration uses a new identity. Reusing an active identity for another configuration is a
conflict. Concurrent incompatible mutations are serialized by `ModelInstanceController`; they
never rely on ACN-side locking. Ready state carries the actual selected parallelism, physical
context allocation, and memory-domain allocation. Hardware snapshots do not own that evidence.
The ICN composition root creates no resident native backend before readiness. Each resident load
creates one private `inference-worker` child; that child initializes its own process-lifetime
native-backend capability, prepares and loads exactly one topology, and owns the executor until it
exits. Persistent ICN exposes the loaded backend through a bounded framed-IPC proxy. Isolated
planner and template workers remain separate metadata-only children with one backend for their
private process lifetime.

Inference-worker lifetime is subordinate to ICN even on abrupt failure. Unix children disable
core dumps and run a dedicated parent-liveness watchdog; Linux additionally requests
`PR_SET_PDEATHSIG`. Windows workers are assigned to a kill-on-close Job Object. The retained child
or Job handle, rather than a later PID lookup, performs forced termination and reaping.

An ordinary chat request names the exact model-instance identity and serving-configuration identity.
ICN atomically acquires `ModelInstanceLease` only when that exact pair is Ready and admission is
open, and holds it until the response stream succeeds, fails, or is canceled. A missing,
non-Ready, or differently configured instance is rejected; chat never starts a model transition.
Before constructing a local-provider request, ACN's `ModelSlotController` observes or loads the
selected slot's exact instance, waits for that ID to become Ready, rechecks the slot binding, and
installs the exact pair in the request fiber. Preparation failure remains outside the provider error
contract.

ACN observes `ModelInstancesSnapshot`; it never derives canonical lifecycle from the load response
stream. The load stream is observational output for the initiating caller only. ACN projects the
matching instance into every slot selecting that configuration. Explicit Stop and autonomous ICN
release therefore produce the same Stopping and Stopped projection on every surface.

Replacing a model must not claim the new model is ready until its backend is usable. Failure leaves
the model instance in an explicitly reported state and must not make requests route to a
half-loaded backend. One Stop endpoint accepts only the exact model-instance identity. During
loading it cancels and cleans partial resources; after readiness it closes mutation admission,
waits for protected inference leases, and releases native resources. Stop is idempotent for an
ended identity and cannot affect a newer occurrence of the same configuration. Chat requests bind
to the active native generation they began with and cannot silently continue on a replacement.

Every inference request holds an exact model-instance lease through stream end or cancellation.
Explicit load, replacement, and Stop share controller mutation authority. Stop and replacement
close new inference admission and wait for existing leases to drain. Memory-pressure
eviction is deliberately different: persistent ICN observes whole-system available memory every
100 milliseconds while a worker exists, and every second while idle. It immediately terminates
the inference worker when availability reaches the configured system reserve. After eviction,
one-second observations must remain above the recovery threshold for the full recovery interval
before load admission reopens. Eviction does not wait for leases or native cleanup. Worker exit, protocol
loss, or unavailable memory supervision terminalizes the affected instance and fails its streams
without terminating persistent ICN. There is no automatic reload.

ICN's pinned runtime is part of the ICN build, so ACN has no separate native-runtime install,
discovery, refresh, instance registry, endpoint lease, or selection lifecycle.

## Failure semantics

Lifecycle errors are typed by phase and preserve a safe cause plus bounded diagnostic evidence:

- resolution and executable verification;
- spawn and startup protocol;
- readiness timeout or incompatible identity;
- unexpected exit;
- graceful-shutdown timeout, forced termination, or reap failure; and
- startup-probe protocol failures mapped from the generated client with their safe cause retained.

Generated client failures distinguish invalid local input, transport failure, a declared remote
failure, undeclared or invalid response, incomplete stream, and cancellation. Declared ICN error
bodies retain their generated type; common failure metadata preserves operation ID and HTTP status.
Lifecycle unavailability/stopping is separate from HTTP protocol failure. Secrets, full model
prompts, authorization values, and unbounded native output are never attached.

Domain results remain values. In particular, `DoesNotFit`, a download `Failed` terminal event, a
safe deletion refusal, and a model load failure reported by the operation contract are not
misclassified as broken HTTP transport. Defects are reserved for violated internal invariants;
expected command, filesystem, HTTP, schema, timeout, signal, and child-exit outcomes remain in
typed Effect error channels.

## Observability

Every child has stable ACN and ICN instance correlation fields. Startup, readiness, requests,
stream termination, unexpected exit, and shutdown create Effect spans and structured logs. Child
stdout/stderr is bounded, line-framed, level-mapped where possible, and correlated with its PID and
instance ID. A diagnostic tail is retained for failures without becoming an unbounded in-memory
log store.

Health, inventory, and hardware observation are not ACN demand. ACN/session leases
cover accepted user operations and inference; ICN backend leases cover native admission and response
streams. These are composed resource gates, not shared activity timestamps.

## Conformance criteria

The lifecycle conforms when:

- one ready ACN has exactly one owned ICN child and no reusable/discoverable ICN daemon;
- constructing `IcnClient` without `IcnProcess` is impossible in the Effect dependency graph;
- ACN cannot become ready when its ICN binary is absent, incompatible, or unready;
- launch is model-free and changing the active model never replaces the ICN process;
- loopback binding has no probe-then-bind race and readiness proves child instance identity;
- every bootstrap record produced by ICN is accepted by its generated Bun schema, and generated
  contract drift fails validation;
- every normalized OpenAPI operation appears automatically in the generated callable client;
- `@magnitudedev/icn` contains no hand-written streaming transport logic or redundant runtime
  facade;
- ACN contains no ICN command service, mutation proxy, or mirror-copy service;
- no Bun implementation duplicates hardware, model inspection, fitting, downloading, inventory, or
  pinned-runtime management;
- the public local provider ID is exactly `local`, and its bound model streams through the scoped
  ordinary generated chat client rather than an endpoint URL adapter;
- chat leases only an exact Ready model instance and never starts a load or replacement;
- ACN slot commands and scoped local-request preparation are the only product paths that start
  explicit ICN load operations; explicit stopping uses only the exact model-instance identity;
- `ModelInstancesSnapshot` drives matching slot projections; the load response stream never owns
  lifecycle;
- loading one local model terminalizes the prior instance before the replacement becomes Ready;
- a resident model remains loaded until explicit replacement, exact-instance Stop, memory-pressure
  eviction, inference-worker loss, or ICN process exit;
- replacement, load, and Stop serialize through native mutation authority and cannot
  invalidate an admitted inference lease;
- product model-download, activation, deletion, hardware, and fit operations reach ICN only through
  the generated client, with no alternate model-repository or host-inspection path in ACN;
- interrupting a consumer stream closes its response without terminating ICN;
- an unexpected ICN service exit causes the owning ACN to fail closed without an in-process
  restart; an internal inference-worker exit unloads only its runtime generation and leaves ICN
  available;
- normal ACN shutdown cancels higher-level work, gives ICN only a short graceful termination window,
  escalates on deadline, and reaps it;
- termination and interrupt signals both activate native graceful shutdown;
- abrupt parent death cannot leave an ICN indefinitely orphaned;
- lifecycle and transport failures remain typed and retain bounded, redacted diagnostics; and
- generated-artifact checks, package tests, native signal tests, and release smoke tests prove the
  shipped ACN and ICN identities are compatible.
