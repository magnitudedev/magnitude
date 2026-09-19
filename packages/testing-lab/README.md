# Magnitude testing lab

This package is under active implementation. The coordinator protocol, persistence, source
transport, provider allocation adapters and initial functional drivers exist. A deployed
scheduler, complete case implementations, image qualification, and CI
workflow are still required before the full target matrix can execute.

Use the Bun version pinned by the root `packageManager` (currently 1.4.2). `bun lab help`
describes the CLI. `bun lab targets` lists the requested coverage; listing a target does not
mean its provider image or backend has been qualified.

```sh
bun lab run --source . --target macos-26-arm64-metal-apple-silicon
bun lab run --source . --profile pr --concurrency 4 --budget 150
bun lab run --artifacts ./dist/release-manifest.json --target macos-15-arm64-metal-apple-silicon
bun lab status --run run-<uuid>
bun lab results --run run-<uuid>
bun lab cancel --run run-<uuid>
```

`LAB_URL` and `LAB_TOKEN` select an authenticated coordinator. The CLI obtains owner/trust
from `/v1/me`, snapshots dirty source and initialized submodules, queries which source objects
are missing, uploads only those objects, registers the immutable input and submits the run.
No commit or push is required. Interrupting the waiting CLI leaves the remote run running;
use `cancel` for cancellation. `--no-wait` returns after submission.

`--artifacts` accepts a release manifest with its application artifacts in the same directory.
Every declared application artifact is copied into local content-addressed storage and checked
against its SHA-256 and byte count before remote submission. The uploaded manifest preserves
release identity and plugin metadata; this mode does not rebuild or publish packages.

`iterate` and `verify` are protocol selections. Warm worker reuse for `iterate` has not yet
been connected. Never present an iteration result as a fresh installer qualification.

The API offers targets, planning, immutable object transfer/input registration, submission,
status, results and cancellation. Authentication supports server-mapped credentials and
Entra/GitHub OIDC. Live Entra acquisition and verification have passed; deployment and an
actual GitHub workflow run remain unqualified.
Owner/trust cannot be escalated by submitting different request fields. PostgreSQL owns run,
work-attempt and lease state with fencing and bounded infrastructure retries. Test dependencies
are ordered per harness; a failed prerequisite blocks dependent cases while independent checks
continue. A missing or blocked check cannot produce a passing overall result.

Azure uses subscription `5304c4b3-d605-4193-b0cb-766c065acfa6`, resource group `magnitude-ci` and
West US 2. Configured image versions must be explicit. VMs have private NICs and delete-on-remove
OS disks. Ownership metadata allows cleanup to discover VM, NIC and disk leftovers even after
an interrupted allocation. The Namespace adapter verifies its catalog timestamp and guest OS
version/build; this is a drift check, not an immutable provider image guarantee. The shared
Spark remains opt-in and has not been exercised during implementation.
The A10 (72 vCPU) and RTX PRO 6000 (144 vCPU) quota requests returned `ContactSupport`;
the portal support request is prepared but awaits required contact details before submission.
Neither GPU family is currently qualified for lab execution.

## Functional probes

These are executable acceptance probes, separate from fixture-backed unit/integration tests:

- `scripts/build-desktop-candidate.ts` uses the release package's existing build/package helpers.
- `scripts/desktop-probe.ts` drives an actual packaged Electron executable, its native bridge,
  navigation, appearance and window lifecycle. `LAB_PROBE_READY=true` also requires service Ready.
- `scripts/generation-probe.ts` downloads and loads a named model through Catalog, then exercises
  real discovery, generation, SSE, tool-call follow-up, invalid inputs and cancellation/retry.
  It selects the model by canonical `LAB_PROBE_MODEL_ID`; `LAB_PROBE_CACHED=true` explicitly
  skips downloading a model already acquired in this isolated profile.

Both UI probes require `LAB_PROBE_ROOT` and `LAB_PROBE_EXECUTABLE`. They retain bounded application
logs, screenshots, Playwright traces and JSON outcomes under that root. Profiles are isolated
with the product's existing test-profile mechanism. Do not upload entire profile directories
as public evidence: they contain disposable application identities and browser state.

The generation probe is not yet a backend attestation or a harness CLI test. Passing it does
not claim Metal/CUDA offload, updater correctness, package-manager installation, or a complete
suite. The final worker must also verify observed hardware/backend receipts and execute every
selected case.

## Development verification

```sh
cd packages/testing-lab
bunx --bun vitest run
bunx tsc --noEmit
```

Database integration tests launch isolated temporary PostgreSQL instances over Unix sockets.
Set `LAB_TEST_POSTGRES_BIN`, provide `pg_config` on PATH, or use the Postgres.app fallback on macOS.
The fixture still requires a Unix host; Windows database tests should use a separate database
service. Provider fixture tests never claim to exercise real cloud VMs.

## Scheduler and harness implementation checkpoint

The durable scheduler now claims target assignments, records ownership before allocation,
heartbeats work and leases, invokes an injected worker runner, and releases resources on success,
failure and cancellation. The independent reconciler cleans expired/cancelled leases and tagged
expired orphans. A finite cleanup timeout leaves a discoverable Releasing lease; it does not mark
the run complete. Real PostgreSQL tests exercise these paths. Local transport uses owned directories,
checks lease markers and rejects transfer traversal and symlink escapes. The actual remote
WorkerRunner and coordinator deployment remain unfinished.

`pi-probe.ts`, `opencode-probe.ts`, and `hermes-probe.ts` exercise product-created connections,
real generation, a read/edit task, and session persistence against a packaged app. Each uses
an isolated harness home, no inherited provider credentials, and a disposable working directory.
They require the generation probe's cached model plus `LAB_PROBE_PI`, `LAB_PROBE_OPENCODE`,
or `LAB_PROBE_HERMES` pointing to the pinned client executable. See `tools/README.md` for versions.

Pi's RPC decoder waits for `agent_settled` and verifies assistant provider/model and successful
terminal generation. OpenCode verifies completed JSON parts and the persisted transcript's
provider/model; its CLI does not emit token deltas, so that alone does not prove streaming.
Hermes verifies streamed text, matching tool starts/results, terminal result and session identity.
Protocol tests inject failures and cannot be mistaken for real inference qualification.

Live macOS results: Pi and OpenCode passed all of those programmatic journeys. Hermes passed
with explicit persistent model selection through the actual bundled CLI. Its fresh UI-only
connection remains a reproduced failure: Hermes's first-run guard ignores the named provider
without a selected default. `LAB_PROBE_HERMES_SET_MODEL=true` with `LAB_PROBE_BUNDLED_CLI` exercises
the separate explicit-selection scenario; it must not replace or hide the fresh-profile case.
Pi and OpenCode terminal interaction also passed against real local generation; Hermes TUI interaction,
complete suite integration, GPU/backend receipts and the full OS matrix remain unqualified. No performance benchmark gates have been introduced.

## UI resilience

Packaged UI tests use the stable action/entity identifiers in `desktop/src/automation.ts`.
They do not locate controls by visible copy, accessible-label wording, icon, color, screen position,
CSS class, or DOM ancestry. Model and harness identity comes from canonical IDs. Catalog search
also accepts a canonical model ID. State assertions use semantic readiness/connection attributes
and accessibility state, followed by actual endpoint/harness behavior; attributes alone do not
qualify generation. Screenshots and traces are diagnostic evidence, not visual golden files.

When redesigning a control, carry its action identifier with it. Keep entity identity on the
containing model/harness component. A real workflow change belongs in the shared DesktopDriver;
individual scenario files should not acquire their own selectors. Removed or unusable actions
must fail clearly. No fallback text selectors, positional selectors, forced clicks, or assertion
retries are used to conceal regressions. Time bounds wait for state, not arbitrary sleeps.

`scripts/ui-resilience-probe.ts` runs the real packaged app with a presentation-only challenge:
rewritten button/heading text and accessible labels, different colors, reversed navigation and
setting order. It exercises navigation, settings, window lifecycle, model loading, Connections and
real generation. This probe passed on local macOS 15.5. It uses the same isolated cached model
profile as the other probes and `LAB_PROBE_TOOLS_PATH` to find Pi. The presentation mutation is
in test support only; it is not shipped as an application feature.

## Native installer and bundled CLI probes

`prepareCandidate` chooses exactly one desktop installer for the target host/format, materializes
its content-addressed bytes, and checks digest and size before publishing the accepted path.
`nativeInstaller` rechecks bytes immediately before invoking native installation. Windows and
Linux mutations require a disposable worker/user; macOS can use a private owned install directory.
Existing installations are rejected. The Windows path uses the real per-user NSIS destination,
and waits for its copied uninstaller process. Linux uses the existing DEB/RPM package managers.
Windows/Linux execution and complete registration, trust and user-data checks remain unqualified.

`install-probe.ts` exercised a real DMG install, exact bundled CLI version and removal on local
macOS 15.5. Its inputs are `LAB_INSTALL_PACKAGE`, `LAB_INSTALL_VERSION`, `LAB_INSTALL_ROOT`, and
`LAB_INSTALL_TARGET`. This narrow probe does not qualify app startup or the entire install suite.

`cli-probe.ts` exercises the packaged binary's version/help/hardware/service commands, cached
model pull/stop/load, real generation after reload, Pi/OpenCode/Hermes add/sync/remove, invalid
inputs and native SQLite/Bun without developer tools on PATH. All nine checks passed locally.
It additionally requires `LAB_PROBE_BUNDLED_CLI`, `LAB_PROBE_VERSION`, and `LAB_PROBE_TOOLS_PATH`.
Public shell registration, interruption and full service lifecycle remain separate work.

Native host inspection is available with `LAB_INSPECT_TARGET=<target-id> bun
packages/testing-lab/scripts/inspect-host.ts`. It checks the selected OS version, CPU
architecture/vendor, and required GPU model. macOS uses native Metal device registry IDs
through a Swift tooling probe (Command Line Tools required on the runner); Windows uses CIM
and NVIDIA queries; Linux uses distribution metadata, lscpu and NVIDIA queries. DGX OS
prefers the installed OTA version over the factory image version. Native Mac execution is
verified; Windows/Linux collectors still require qualification on their actual workers.
This observation does not prove that inference used the selected backend.

The initial guest-side artifact worker connects manifest integrity, host inspection, exact
installer download, native installation, packaged launch/readiness, bundled CLI checks and
scoped uninstall. Unconnected cases return blocked outcomes. The scheduler now transfers
owner-authorized immutable inputs to this worker and verifies the returned attempt, case set
and evidence hashes. Guest execution receives no coordinator or provider credentials.
Source compilation and packaging now use the guest worker path; guest runtimes must be explicitly configured
for each provider/artifact host and are not automatically installed on stock cloud images.
`artifact-worker-probe.ts` is an explicitly limited five-case install/CLI diagnostic, not a
full-profile run; it has passed against the local macOS 15 DMG with verified removal.

`scheduler-worker-probe.ts` runs a full quick-profile plan through PostgreSQL, the scheduler,
a local lease, the actual guest subprocess and native installer. It preserves all required
cases; the current incomplete worker returns failures/blocks, not a narrowed green profile.
Its environment inputs match the artifact-worker probe (`LAB_WORKER_ROOT`,
`LAB_WORKER_MANIFEST`, `LAB_WORKER_TARGET`). Failure diagnostics are content-addressed
and transferred before the scheduler deletes the owned worker.
Finalized Playwright traces and bundled-CLI command logs are also exported to content-addressed
storage. UI traces are attached to the launch case and CLI logs to the CLI version case. Export
failures remain visible as cleanup errors while preserving the original test results; files outside
the owned evidence directory, symlinks and oversized diagnostics are rejected.

## Source build execution

Source workers extract admitted objects into a fresh workspace and use `bun install --frozen-lockfile`
under the pinned runtime. Compilation and packaging invoke the existing release helpers in separate
processes. The compile receipt binds source digest, commit and native host; packaging emits the
release manifest (including the Mac update ZIP). Artifact admission verifies all output bytes before
installation. Compiler/packager failures remain separate case outcomes and preserve command logs.
The worker preserves explicitly configured `CARGO_HOME` and `RUSTUP_HOME` while isolating the
build's HOME. Bun compilation uses the release-owned host target, including the Linux x64
baseline target.

`scripts/source-build-probe.ts` exercises these phases on a real local source snapshot. Set
`LAB_BUILD_PROBE_ROOT` to a new directory and `LAB_BUILD_PROBE_TARGET` to the actual local target.
The lower-level candidate script accepts `LAB_BUILD_PHASE=compile|package|all`, `LAB_BUILD_OUTPUT`,
`LAB_BUILD_SOURCE_DIGEST`, and `LAB_BUILD_SOURCE_COMMIT`.

An actual Azure Ubuntu 24.04 Intel x64 source worker passed compilation and production of both
existing DEB and RPM formats. The exported installer, service and CPU inference archives were
independently admitted by hash and length; the producer VM, NIC and disk were removed. Building
an RPM on Ubuntu does not qualify installation on Fedora or Red Hat.

A separate clean Ubuntu consumer passed 25 functional checks on those same artifacts: native
installation, packaged UI/model download, settings persistence, CPU-attested generation, streaming,
tool-result follow-up, cancellation/reload/restart, bundled CLI and native removal. The dependency
audit exposed two lab assumptions: the launcher is a shell script, and Electron also needs the
OS-owned `libudev`. After correction, that audit passed on another fresh consumer, including
native symbol-version and package-owner checks. These are separate recorded attempts, not one
all-green run. Linux signatures, harnesses, updates and the other distro targets remain unqualified.

Scheduled source execution still builds on the allocated target before installation. Separate
producer/consumer orchestration, shared producer deduplication and native Windows toolchain
configuration remain required for full remote verification. The shared worker already serves
admitted private runtime archives through its scoped release fixture.

## Current packaged-app execution

The candidate worker connects catalog search and details, a fresh UI model download, UI model
loading, endpoint discovery, nonstreamed/streamed/tool generation, invalid requests and cancellation.
Generation cases still depend on backend attestation; they remain blocked until the live observation
adapter is connected. A capability inventory does not qualify CPU/Metal/CUDA execution.
CLI readiness depends on successful app readiness, so a failed owning service blocks that dependent
check instead of repeating the readiness timeout.

`namespace-worker-probe.ts` exercises the real transport against an explicitly named, already
lab-tagged Namespace Mac and releases it afterward. It needs `LAB_NAMESPACE_NAME`,
`LAB_NAMESPACE_CLI`, `LAB_GUEST_BUN`, `LAB_GUEST_ENTRY`, and the three artifact worker variables.
The guest runtime and its dependencies must already be installed. This is a provider qualification
probe, not production coordinator deployment or a replacement for scheduled allocation.

A current source snapshot built a DMG and update ZIP successfully on Namespace macOS26.6.2.
The exact DMG passed local macOS15 launch/readiness/CLI checks and a fresh UI acquisition/model-load
run (11 passed,11 blocked,zero failed cases,clean removal). The cloud app launched but its inference
process exited before readiness; cloud generation remains unqualified. Provider cleanup completed;
the empty-inventory CLI notice exposed and now has a tested parser fix. No borrowed Mac remains.

Package identity now inspects installed Mach-O, ELF64 or PE32+ headers and checks the running
desktop against the exact bundled service and CLI versions. The native host and Unix command
helper must also contain the requested architecture. This is distinct from dependency and
production-signature validation, which remain separate cases. `install-probe.ts` records native
host identity, installs the exact package, launches its desktop, checks payload identity and removes
the owned installation even when assertions fail. This path passed with the current DMG on
macOS15.5 ARM64; Linux and Windows header fixtures do not qualify native installation there.
The corrupt-candidate case changes one byte in a same-length private installer copy and requires
the integrity rejection before native package commands. It preserves the admitted original and
removes the corrupt copy. The real macOS probe passed this rejection followed by normal installation,
package identity verification and removal. Scheduled CLI coverage also connects catalog list/show,
cached pull, status, stop and reload through the bundled binary.

Connections acceptance seeds an unrelated provider in the isolated harness home, then uses the
app to connect, refresh, disconnect and reconnect each selected harness. It checks preservation,
provider removal, the exact app endpoint and records the final configuration digest. Quick runs
exercise Pi; broader profiles exercise Pi, OpenCode and Hermes. `connections-probe.ts` qualified
all three UI flows locally against a packaged app with cached model bytes; this does not qualify
their generation on other operating systems. The probe requires the usual `LAB_PROBE_ROOT`,
`LAB_PROBE_EXECUTABLE`, `LAB_PROBE_TOOLS_PATH` and an isolated profile with a downloaded model.

The worker connects H1–H6 to real Pi/OpenCode/Hermes processes, with shared generation evidence,
conversation identifier recall, bounded read/edit fixtures and persisted-session reuse. Images must
configure absolute `LAB_PI_EXECUTABLE`, `LAB_OPENCODE_EXECUTABLE`, `LAB_HERMES_EXECUTABLE` paths;
the suite rejects versions outside the pinned tools set. Native event logs are retained per harness.
Version checks alone do not qualify an immutable worker image. H7 dispatches Pi and OpenCode terminal journeys;
Hermes H7 remains explicitly blocked until its terminal journey is implemented.
Worker images must also configure an absolute `LAB_TERMINAL_NODE_EXECUTABLE` for Node 24+.

`TerminalDriver` owns a Node.js 24+ subprocess hosting `node-pty`, while the Effect worker
interprets its output with `@xterm/headless`. It launches argument arrays with an explicit
environment, reports the native child's exit, supports keyboard input and resize, bounds output,
and retains raw/parsed terminal evidence. Runtime and harness executable paths are explicit.
The lab-only `node-pty` pin is `1.2.0-beta.15`: the stable 1.1.0 tarball has a documented
[macOS executable-permission defect](https://github.com/microsoft/node-pty/issues/919).
The selected package includes Unix and Windows native bindings; these are not app runtime assets.

Set `LAB_TERMINAL_NODE_EXECUTABLE` to an absolute real Node executable when running
`test/terminal.test.ts`; Bun's `--bun` launcher can otherwise put a Bun shim ahead of Node.
Five native Mac ARM64 fixtures passed: rendered cursor/Unicode/alternate-screen behavior with
resize and keyboard interruption, forced cleanup, cancellation cleanup, missing-executable
rejection, and output-bound failure cleanup. Node 26.8.1 and Bun 1.4.2 were used for this check.
Linux and Windows terminal qualification remain pending. Passing these fixtures does not establish
harness generation or cancellation.

With `LAB_PI_EXECUTABLE` additionally pointing to Pi 0.85.1, `test/pi-terminal.test.ts`
exercises the real Pi TUI against a controlled loopback SSE endpoint: keyboard model selection,
rendered streaming output, Escape cancellation confirmed by both the aborted request and Pi's
persisted assistant record, a successful follow-up, and normal keyboard exit. This passed on
Mac ARM64. It qualifies terminal automation against the pinned client, not Magnitude generation.
A second fixture rejects a first turn that finishes naturally before Escape; keyboard input alone
cannot satisfy interruption. The fixture waits
through Pi's startup using only an idempotent model-selection command and never retries generation.

`test/opencode-terminal.test.ts`, with `LAB_OPENCODE_EXECUTABLE` set, exercises OpenCode
1.18.31's real terminal against synthetic SSE. It selects a model and off variant by keyboard,
observes streamed output, performs the pinned two-Escape interruption, validates the exported
assistant abort record, completes a follow-up and exits normally. Its negative fixture rejects
naturally completed generation before interruption. Model labels come from the app-created
connection; canonical session/provider/model IDs in the native export remain the assertions.
The keyboard contract is verified against the [pinned OpenCode source](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/tui/src/component/prompt/index.tsx).
Failure screens are captured before terminal cleanup restores the alternate screen.

`harness-terminal-probe.ts` uses the packaged app's UI-created connection and the same reusable
journeys as H7. Both Pi and OpenCode passed on local Mac ARM64 with Qwen3.5 4B Q4: rendered
partial generation, a persisted aborted assistant turn, successful follow-up in the same identified
session, and normal keyboard exit with no cleanup errors. It takes `LAB_PROBE_HARNESS` (`pi` or
`opencode`), `LAB_PROBE_ROOT`, `LAB_PROBE_EXECUTABLE`, `LAB_PROBE_MODEL_ID`, the corresponding
`LAB_PI_EXECUTABLE` or `LAB_OPENCODE_EXECUTABLE`, `LAB_TERMINAL_NODE_EXECUTABLE`, and optional
`LAB_PROBE_PORT`. This standalone probe does not qualify the complete worker matrix or backend;
regular H7 retains the existing H2/backend prerequisites and exports session, screen and terminal
artifacts before a guest is removed.

`harness-suite-probe.ts` exercises these scenarios independently of backend qualification. It accepts
the Connections probe variables plus `LAB_PROBE_MODEL_ID` and optional comma-separated
`LAB_PROBE_HARNESSES`. Local Pi passed generation, recall, exact editing and persisted recall.
OpenCode generated and recalled the identifier but failed to perform the requested file tools;
its JSON adapter also cannot attest token streaming. Fresh-profile Hermes rejected the app-created
connection during first-run setup before generation. These live failures are retained, not converted
into passes by injecting provider settings or retrying assertions.

`azure-lifecycle-probe.ts` exercises the real allocator on a disposable Ubuntu x64 Intel VM:
allocate the same lease twice, run a bounded native guest identity command through Azure's agent,
and release only that lease's resources even when the probe fails. Set `LAB_AZURE_CONFIG` to a
validated AzureConfig JSON file, `LAB_AZURE_PROBE_ROOT` to a diagnostic directory and
`LAB_AZURE_TARGET=ubuntu-24.04-x64-cpu-intel`. The configured subnet must already exist in the
same subscription/resource group. This probe owns its VM/NIC/disk, not the shared subnet. It does
not qualify app installation, artifact transfer, the full guest bootstrap or GPU execution.
Full workers also require an explicit outbound route, such as a NAT gateway, for package
repositories, admitted artifact downloads and model downloads. A private NIC alone does not
establish Internet egress. The allocator does not modify shared network policy or attach public
IPs to workers; provision and account for the subnet's egress separately.
The live probe passed in the credited subscription's `magnitude-ci` group in WestUS2 with
Standard_D4s_v6 and Canonical Ubuntu24.04 image24.04.202609040. The VM had no public IP;
the guest reported Ubuntu24.04, x86_64 and GenuineIntel. VM, disk and NIC cleanup passed.

Azure Blob storage is available behind the same ArtifactStore interface. The coordinator uses
Entra authentication (`--auth-mode login`), stages bounded files, verifies SHA-256 before upload
and on download, publishes each content address once and pins downloads to the observed ETag.
Workers do not receive account keys. The current implementation uses the installed Azure CLI;
high-volume transfer throughput and outward worker bootstrap remain to be qualified.
`azure-artifact-probe.ts` accepts `LAB_AZURE_ARTIFACT_CONFIG`, `LAB_ARTIFACT_FILE` and
`LAB_ARTIFACT_REPORT` and verifies a real round trip. It leaves the immutable private object in
the configured container. The live 201335712-byte Mac DMG round trip passed in the credited
subscription's private `magnitudelab5304c4b3/artifacts` storage. No compute resource is retained.

## Coordinator service

Run `bun lab serve` with the pinned Bun runtime, `LAB_COORDINATOR_CONFIG` pointing to a JSON
configuration and `LAB_DATABASE_URL` holding the PostgreSQL connection string. Secrets are read
from named environment variables, not embedded in the configuration or printed at startup.
A minimal local API configuration is:

```json
{
  "coordinator": { "instance": "local-dev", "concurrency": 2, "accountBudgetUsd": 100, "pollMs": 1000, "reconcileMs": 10000 },
  "hostname": "127.0.0.1",
  "port": 11399,
  "credentials": [{ "tokenEnvironment": "LAB_DEVELOPER_TOKEN", "principal": { "owner": "developer", "trust": "developer" } }],
  "storage": { "kind": "file", "directory": "/absolute/path/to/lab-objects" },
  "runtimes": []
}
```

Supply a distinct credential of at least 32 characters per configured identity. The server runs
migrations before accepting HTTP work, shares durable stores with scheduler/reconciler loops,
and keeps cleanup scoped to its leases. Empty runtimes/providers support API setup and return
explicit blocked run results; they cannot qualify an app test. Namespace configuration requires
both pinned image observations and an installed guest runtime. Azure Linux can use the outward
worker protocol through the optional `azure` configuration described below. Spark execution
and fresh guest runtime provisioning are not yet configured by this entry point. Azure Blob storage can replace
the file storage using `{"kind":"azure","config":<AzureArtifactConfig>}`.

For remote deployment, terminate HTTPS at the ingress; the bearer credential mode is the initial
private-service path. Entra/GitHub OIDC and outward worker polling are implemented; Bicep
deployment and managed coordinator hosting remain unfinished. A real HTTP/PostgreSQL test covers upload, admission,
idempotent submission, automatic scheduling, results, owner isolation and configured startup.

GitHub OIDC admission can be enabled with a `github` configuration containing `audience`
and `repositories: [{ repositoryId, ownerId }]` (immutable GitHub numeric IDs as strings).
`credentials` may be empty when GitHub is configured. Tokens must use the configured audience;
all admitted GitHub runs receive `untrusted-ci` permissions, isolated by run ID and attempt.
Supported events are pull_request, push, workflow_dispatch and merge_group. This is the server
verification boundary. In GitHub jobs set `LAB_AUTH=github`, `LAB_OIDC_AUDIENCE` to the
configured audience, and grant `id-token: write`; omit `LAB_TOKEN`. The CLI uses the runner
provided token endpoint and renews its cached credential after one minute. A deployed GitHub
workflow qualification remains pending. Live Entra acquisition and verification have passed. See [GitHub OIDC claims](https://docs.github.com/en/actions/reference/security/oidc)
and [JOSE verification](https://github.com/panva/jose).

Developer identity authentication uses `LAB_AUTH=entra`, `LAB_ENTRA_TENANT` and
`LAB_ENTRA_APPLICATION` (UUIDs), with Azure CLI already signed in. Omit `LAB_TOKEN`.
The server's `entra` configuration contains `tenantId`, `applicationId`, and `users`
(an explicit array of Entra object IDs). The API application must issue v2 access tokens
and expose the delegated `Lab.Access` scope under `api://<applicationId>`; Azure CLI
needs consent for that scope. The client renews through Azure CLI without printing tokens.
Only allowed users with that delegated scope receive developer permissions. ARM tokens,
ID tokens and app-only role tokens do not qualify. The `magnitude-testing-lab` registration is provisioned in tenant
`4581d4bf-a664-4a42-a66a-c842beeec9e7`, application
`b47912ec-a3bb-49f4-ac37-25e06b4f7743`. Azure CLI is preauthorized only for
its `Lab.Access` scope; no client secrets or certificates exist. Real Azure CLI
acquisition and Microsoft JWKS verification passed for Tom's allowed object ID
`7b68fd28-7906-4529-aab6-c550148572f1`. This qualifies authentication, not remote
worker execution or the full target matrix.
See [Microsoft's claims validation guidance](https://learn.microsoft.com/en-us/entra/identity-platform/claims-validation).

Recheck developer authentication with `scripts/entra-auth-probe.ts` using
`LAB_ENTRA_PROBE_CONFIG` containing the server's `entra` JSON configuration.
The probe prints the admitted principal only, and does not save credentials.

A4 now changes appearance through stable UI action IDs, closes the packaged app,
restarts with the same isolated profile, verifies the persisted choice, repeats for
a second choice, and checks service readiness. Each launch retains separate trace
and process evidence. `scripts/settings-relaunch-probe.ts` exercises the same session
lifecycle locally using `LAB_PROBE_ROOT` and `LAB_PROBE_EXECUTABLE`. The real packaged
macOS 15 ARM64 probe passed both dark/light persistence and final readiness, with no
cleanup errors. This does not qualify Windows, Linux or macOS 26.

A6 exercises sidebar and native window hide/reopen behavior, requests normal Electron
quit, requires exit code zero without a termination signal, then relaunches and checks
service readiness. Forced cleanup is never a passing quit result. The local settings
relaunch probe also runs this lifecycle case; traces close before quitting and logs
remain available after the process exits.
The combined A4/A6 packaged macOS 15 ARM64 probe passed with zero cleanup
errors at `/tmp/ml-app-lifecycle-fixed-20260918/report.json`; all four launch
traces were saved and no probe application process remained.

C4 now runs the bundled CLI connection lifecycle for the selected harnesses and
inspects actual configuration after add, sync, remove and reconnect. It shares
the isolated unrelated-provider fixture with A5. The real macOS 15 ARM64 bundled
CLI passed for Pi, OpenCode and Hermes with preserved unrelated configuration and
the expected endpoint. Evidence: `/tmp/ml-cli-connections-model-20260918/cli-connections-report.json`
and its `cli-evidence` directory. `scripts/cli-connections-probe.ts` requires an
installed model in its isolated profile because CLI model selection validates that
prerequisite. These connection checks do not qualify harness generation.

The harness-error scenario faults only an existing isolated configuration file,
requires a visible error and file-specific repair guidance, checks that malformed
bytes remain untouched, restores the original bytes even if the assertion fails,
and reconnects through the app. `scripts/connections-probe.ts` includes this path
for Pi, OpenCode and Hermes. A7 now combines this path with occupied-port service failure and recovery after
relaunch. It requires an installed test model, because harness connection configuration
requires model availability.
The independent live Mac error checks passed Pi and Hermes but found missing
file-specific repair guidance for OpenCode. The failing trace is preserved at
`/tmp/ml-connection-errors-all-20260918/evidence/ui-trace.zip`. This is a
recorded acceptance failure, not a qualified A7 result.

The real occupied-port service failure/recovery probe passed at
`/tmp/ml-service-error-20260918/report.json` with zero cleanup errors. A7 is wired
to the worker, but full A7 acceptance is not claimed: OpenCode's diagnostic remains
a known failure, and the combined worker scenario has not yet been qualified on
the complete target matrix. No test fault terminates an existing port owner.

R2 removes the test model in a disposable guest, starts a new download through the UI, and
waits for positive incomplete byte progress before interrupting external networking. It requires
the app to show acquisition failure, restores networking, retries through the UI and independently
stream-hashes every target and companion file against the admitted catalog. Successful recovery
also requires backend-attested generation and unchanged app/service ownership. Preflight proves
network isolation and restoration before removing the model. Partial receipts survive failures.
Stable acquisition-state and progress attributes keep these checks independent of button copy,
color and layout. The Electron fixture exercises presentation changes.

The clean Azure Ubuntu 24.04 Intel run at `/tmp/ml-download-recovery-20260919/verified-result.json`
passed all 14 selected cases, including R2 and R3. R2 observed 16,375 of 3,666,233,216 bytes
before isolation, required the rendered download failure, retried through the UI, verified both
model and projector hashes, and generated `HELLO` with the admitted CPU module. App/service
ownership was unchanged. R3 then reloaded into a new worker and generated offline. Both native
network fixtures passed; uninstall and cleanup reported no errors. The packages came from a
separate, verified native source producer. This qualifies this Ubuntu/Intel scenario, not the
remaining distro, GPU or operating-system matrix.

R3 proves cached offline generation by establishing a fresh external TCP control, cutting the
disposable user's external traffic, stopping/reloading the cached model through the bundled CLI,
and attesting generation on a new native worker. It verifies the cut again after generation and
restores external connectivity while preserving app/service ownership. Failed generation still
releases isolation and records restoration; partial evidence and cleanup failures remain visible.

Linux uses a private nftables `inet` table scoped to the qualified guest UID, preserving loopback
and other users. TCP rejection uses a reset so established downloads fail promptly; other
external protocols are rejected too. Workers require `/usr/sbin/nft` and noninteractive sudo for it. The implementation
preflights both creation and idempotent removal; it never flushes a shared table. See the
[nftables manual](https://netfilter.org/projects/nftables/manpage.html) for UID matching and table
lifecycle. `network-fault-native.test.ts` requires `LAB_NATIVE_NETWORK_FAULT=1` on a disposable
Linux guest; it checks external rejection, loopback availability and cleanup after success and
an injected failure. This opt-in must never be set on a shared developer machine or Spark.
macOS and Windows isolation are not yet implemented.

The clean Azure Ubuntu 24.04 Intel run at `/tmp/ml-offline-recovery-20260919/verified-result.json`
passed all 13 selected cases, including R3. The cached model reloaded from worker generation 1
to 2 and returned `HELLO` offline with matching admitted CPU-module bytes. App/service ownership
was unchanged, external connectivity was restored, and the four bundled CLI commands were
exported despite C1 not being selected. Both native isolation/cleanup fixtures passed. Other
distributions and GPU targets remain unqualified for this scenario.

R4 generates and attests the requested backend, terminates only the verified resident
inference worker, observes the model's failed state, explicitly reloads, then generates
and attests again. The native worker generation must change while the application,
service and persistent inference server remain the same. Before/fault/after receipts
survive failed assertions. Linux uses pidfds after checking the installed executable,
worker role, user and service ancestry; macOS and Windows termination remain blocked
until their native mechanisms are implemented and qualified. Unit coverage does not
qualify a native crash/recovery journey.

The clean Azure Ubuntu 24.04 Intel run at `/tmp/ml-worker-recovery-egress-20260919/verified-result.json`
passed all 13 selected package/install/app/generation/recovery/removal cases. Worker generation
changed from 1 to 2 and both generations returned `HELLO` with the admitted CPU module; app,
service and persistent ICN identities were unchanged. The native pidfd fixture also passed refusal
of a foreign profile, parent and service PID. This qualifies the Linux CPU journey on that guest;
CUDA, other distributions and other operating systems still require native qualification.

R5 is connected: bundled CLI stop must reach `Unloaded`, reload must reach `Ready`,
and a subsequent endpoint generation must complete. The real packaged macOS 15 ARM64
probe passed at `/tmp/ml-model-reload-20260918/reload-report.json`, recording separate
completion IDs before and after reload plus all CLI output. The diagnostic does not
attest the selected hardware backend; scheduled R5 remains gated on E2/E6 evidence.
`model-reload-probe.ts` uses the same isolated model/cache setup as the connection probe.

R6 reads the native application/service PIDs and service instance identity, issues
repeated bundled CLI starts, and requires stable ownership. Across two app restarts
it verifies that each previous service process has exited and that the replacement
has a new identity. The real macOS 15 ARM64 diagnostic passed six start requests
across three app lifetimes at `/tmp/ml-service-ownership-20260918/ownership-report.json`.
All observed app/service PIDs were gone after cleanup. This does not qualify other
OS targets or unobserved auxiliary processes.

Installation ownership now supports explicit remove/reinstall through a serialized
lifecycle, avoiding duplicate uninstall during final cleanup. The native macOS 15
ARM64 uninstall/reinstall diagnostic passed at
`/tmp/ml-uninstall-reinstall-20260918/uninstall-report.json`: app and CLI removed,
isolated user data retained, theme preserved after reinstall, service ready, and
final cleanup removed the replacement app. `scripts/uninstall-probe.ts` uses the
same native installer and lifecycle as the worker. Payload verification detects
dangling CLI symlinks as leftovers. This is partial uninstall qualification:
startup registration and X2 remain unfinished; scheduled X4 stays blocked by X2.

X1 is connected to explicit native removal and publishes a removal receipt. Linux
checks the package database and desktop launcher; Windows checks the uninstall
registry key, Start menu shortcut and user PATH registration. Inspection errors
fail verification rather than implying absence. Mac payload removal has native
probe evidence; Linux/Windows registration checks currently have fixture coverage
and still require native qualification. Explicit X1 removal is tested to avoid a
second uninstall during worker cleanup.

X3 captures the complete stopped application's isolated profile before native removal
and compares it afterward: file hashes, directory entries and symbolic links. It does
not follow links outside the profile or put file contents into evidence. X4 reinstalls
the same candidate, verifies its running version and retained appearance, and waits
for service readiness. Its X2 prerequisite remains enforced. The native macOS probe at
`/tmp/ml-retained-profile-20260918/uninstall-report.json` passed full-profile retention
and reinstall, with no cleanup errors. This is not login-startup or cross-platform
qualification. Isolated product profiles intentionally disable login startup; that
case needs its own disposable native-user execution path.

## Local selection and CI reports

`bun lab run --source . --target ubuntu-24.04-x64-cpu-intel --suite app,cli --harness pi,hermes`
submits the current unpublished source with those suites and their prerequisites.
Custom selection requires explicit targets (comma-separated for multiple targets),
replaces profile selection, and defaults to all three harnesses. Empty or duplicate
selection entries are rejected. `--profile quick --harness opencode,hermes` replaces
Pi while preserving quick's case selection. PR/full/release cannot be narrowed with
that flag; use an explicit custom suite selection.

Add `--json out/run.json --junit out/junit.xml` to a waiting run, or use these flags
with `bun lab results --run RUN_ID` to export a completed run. Output directories are
created and each file is replaced atomically. Reports cannot accompany `--no-wait`.
JSON preserves the complete result. JUnit records product failures as failures and
blocked/cancelled/not-selected required cases as errors. Missing/duplicate results,
unselected result entries and cleanup failures also become errors, so CI cannot
mistake incomplete coverage for success. Evidence digests and paths accompany each
case. Existing command exit-code semantics remain authoritative.

## Outward worker access (in progress)

The coordinator now serves `/v1/worker/assignment` using an attempt-scoped opaque
credential, separate from developer/CI authentication. Issuance verifies the durable
assignment, stores only a token digest and allows one credential per attempt. Each
request checks the current claim/fence, run deadline and state. Cancellation,
completion, revocation and reassignment deny access immediately; responses are
noncacheable. Native provider credentials are not included in the invocation.

Azure Linux bootstrap and scheduler integration are implemented as described below;
qualified guest images and complete cloud app execution remain outstanding. Namespace
continues using its existing transport path. PostgreSQL plus live HTTP tests cover credential persistence and
invalidation; no Azure worker qualification is claimed by those tests.

Worker input downloads are now available at `/v1/worker/objects/:digest`. They permit
only the assigned manifest and its referenced source/artifact objects; other
same-owner uploads remain inaccessible. Live HTTP/SQL tests exercise both input
kinds and credential invalidation. This does not yet enable Azure execution.

Worker input graphs use a bounded, owner-scoped immutable cache (eight manifests,
five-minute lifetime). Concurrent object downloads share manifest parsing. Every
request still checks live credential authority, and failed graph reads are evicted
immediately. This avoids rereading a large source manifest for every source file.

`POST /v1/worker/result` now persists an immutable attempt receipt. It validates the
claim, exact selected-case membership and attempt-specific evidence metadata.
Identical redelivery is idempotent; changed replies return conflict. Authorization
and receipt insertion share a transaction with locks on the live attempt/run.
Receipt storage does not finish a run or bypass allocator cleanup. Runner polling
integration is still unfinished.

`PUT /v1/worker/evidence/:digest` requires an exact Content-Length and reserves
storage before consuming bytes: at most 256 MiB per object, 4 GiB and 1,000 objects
per attempt. Both active and verified uploads count toward those limits. Uploads
have a ten-minute deadline; abandoned reservations expire after fifteen minutes.
Only matching SHA-256 and byte length can transition an upload to verified, with
a fresh authorization check before publication. Failed uploads release their own
reservations. Existing verified bytes may be redelivered, but are checked again.

The guest HTTP client now supports assignment fetch, hash-verified streamed downloads,
length-declared evidence upload and result submission. It accepts HTTPS origins or
loopback HTTP for local tests, rejects redirects, bounds transfers and never retries
mutations automatically. Live protocol tests use this client against the real HTTP
API and temporary PostgreSQL database.

`src/outward-worker.ts` is now an executable guest entry point configured by
`LAB_URL`, `LAB_WORKER_TOKEN`, and `LAB_WORKER_ROOT`. It downloads admitted inputs
into a fresh owned workspace, invokes the same native executor as the transport
worker, saves `reply.json` before delivery, uploads unique evidence and returns the
result. It checks live assignment authority every ten seconds and honors the run
deadline. Existing workspaces cannot trigger another execution. Failed delivery
leaves the reply available for inspection. Set `LAB_WORKER_ACTION=deliver` with the same
workspace, origin and still-live attempt credential to deliver the saved result without
running the native executor. The default action remains `execute`, which rejects an existing
workspace. Recovery checks the complete saved invocation against live authority, bounds
saved JSON documents to 16 MiB, validates the reply, and rechecks every evidence hash/length.
It cannot recover after the attempt is revoked, completed or expired. No top-level coordinator
resume operation or automatic provider relaunch is claimed by this guest entry point.

The native recovery probe (`LAB_PROBE_DELIVERY_RECOVERY=true`) deliberately rejected its
first result submission after packaged app cleanup, then delivered through the real HTTP and
PostgreSQL path without providing a native executor. Its reports are
`/tmp/ml-outward-redelivery-20260918/delivery-recovery.json` and `outward-report.json`:
seven cases passed, P4/P5/I4 remained blocked, and cleanup errors were empty. This is a
local macOS package/install recovery probe, not Azure or GPU qualification.

The real macOS 15 ARM64 artifact probe in `scripts/outward-worker-probe.ts` passed
its install/launch/version/corrupt-installer checks through a loopback HTTP coordinator,
including evidence upload and result receipt. Report:
`/tmp/ml-outward-native-20260918/outward-report.json`. Seven cases passed and P4/P5/I4
remain explicitly blocked. Cleanup errors were empty and the installed app was gone.
P1/P2 recorded artifact provenance/verification; they did not compile or package anew.
This does not qualify Metal inference, other platforms or Azure execution.

`outwardWorkerRunner` implements the scheduler's worker interface: it validates the
allocation, issues a scoped credential, calls a configured provider bootstrap, polls
the receipt and revokes the credential on every exit path. A revocation failure adds
a cleanup error without replacing valid case results. The scheduler still owns machine
release. The native probe now exercises this runner with a local bootstrap; its latest
report is `/tmp/ml-outward-runner-captured-20260918/outward-report.json` (seven passed,
P4/P5/I4 blocked, no cleanup errors, test app removed).

The configured coordinator retains Namespace transport and accepts an optional Azure entry:
`"azure": { "allocation": <AzureConfig>, "workerOrigin": "https://your-coordinator" }`.
Azure runtimes use provider `azure`, an absolute guest executable/root and Linux artifact host.
The image must already contain the pinned worker runtime, build/package tools and display
setup; configuring an ordinary marketplace image alone does not supply these dependencies.
The coordinator validates VM identity and exact lease tags, then delivers the attempt credential
through a managed Run Command protected parameter. The request body exists only in a scoped
0600 temporary file; the token is absent from CLI arguments and script text. The script runs as
the configured guest user, and the guest returns results through HTTPS. Delivery is bounded by
the lease deadline and is never automatically replayed after an ambiguous response.

Azure Linux bootstrap and server wiring have targeted tests, including executing the generated
shell with hostile-looking argument literals. A live Ubuntu 24.04 Intel VM also verified delivery:
`/tmp/ml-azure-bootstrap-fixed-20260918/report.json` and `bootstrap.json` record matching protected
credential, configured guest user, workspace and origin, followed by clean VM/disk/NIC removal.
The first live attempt exposed that Azure's `runAsUser` drops named protected parameters through
sudo. The bootstrap now receives the credential in the agent context and explicitly switches to
the guest user while preserving only the three lab environment variables. No credential value
is inserted into command arguments or script text. This qualifies bootstrap delivery on that
Ubuntu image only; complete outward app execution and other Linux images remain unqualified.
Windows is rejected by this bootstrap and requires an interactive-session launcher; a VM agent
service-session launch cannot qualify the desktop suite. Microsoft's parameter semantics are
specified in [Managed Run Command](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/run-command-managed).

## Bundled CLI interruption

C5 now runs invalid-command exit checks and interrupts the actual bundled CLI while it waits
on its owning service. The Unix driver pauses only the service PID observed through the
isolated desktop, waits for the CLI's established loopback connection, sends SIGINT to its
owned CLI process group, and requires interruption exit behavior. The service resumes on
success, failure and cancellation. The case then checks CLI inspection and unchanged native
application/service identity. Invalid commands must exit 1 with a diagnostic; arbitrary
nonzero exits cannot satisfy that check. C5 now depends on A1 as well as C1.

The packaged macOS ARM64 probe passed at
`/tmp/ml-cli-interruption-exit-20260918/interruption-report.json`: exit 130, unchanged owning
identities, usable service and no cleanup errors. Three tests exercise service resumption on
observer failure, early CLI exit and cancellation. Unix guest images need `lsof`; Linux native
qualification remains outstanding. Windows requires a native console driver and stays blocked
for this case. This does not claim cancellation of model acquisition or generation.

## Update UI driver

The desktop driver now exposes Settings update actions (check, download, restart, discard),
automatic-download control and transfer-state waits with candidate version observations.
Selectors use stable action IDs and rendered domain-state attributes. A real Electron renderer
fixture exercised these controls with deliberately changed labels/colors and verified immediate
failure reporting. This fixture does not update an installed app or qualify U1–U6. Old/new package inputs and private acceptance routing are implemented; a signed fixture service
and native suite orchestration remain. Production trust remains a separate qualification.

Five targeted driver/session/error tests passed and testing-lab typechecking passed. The desktop
package typecheck returned exit 2 with Effect diagnostics in unchanged main/preload and dependency
files, including `multipleEffectProvide` at desktop/src/main.ts:146; no renderer/automation diagnostic
was emitted. Full output: `/tmp/ml-desktop-update-typecheck-20260918.log`.

## Previous-release input for update testing

`bun lab run --source . --update-from /path/to/old/release-manifest.json --target <target> --suite update`
freezes the previous release alongside the unpublished source. `--artifacts` may replace
`--source` for an already packaged candidate. The API request uses optional
`updateFrom: { kind: "artifacts", digest: "<manifest sha256>" }`.

Both manifests must be registered by the caller before admission. Missing baseline registration
is rejected. Namespace transport and outward HTTP workers transfer both verified graphs; worker
credentials do not gain access to other owner uploads. Tests exercise baseline graph download,
revoked/missing access, native-executor handoff and real HTTP/PostgreSQL admission rejection.
The shared worker executes U1 from this input: it validates the package pair before native changes,
installs the previous package into a separate profile/control directory, verifies desktop/service/CLI
identity and persists appearance across an actual application/service restart. It then closes the
baseline and restores the primary package ownership. Missing baseline blocks U1; an invalid pair
fails before changing the installation. U2–U6 worker integration remains outstanding.

Update pair preparation now verifies and materializes the previous native installer, the
candidate native installer, and the candidate update archive before installation changes.
Candidate version must be newer; identical installer bytes cannot represent different versions.
Both releases must contain exactly one installer matching the target host and package format.
macOS additionally needs exactly one matching ZIP, while Windows/Linux update with their native
EXE/DEB/RPM. Content-address verification and byte counts apply to all materialized files.
Seven pair/candidate tests passed across these package formats. These are artifact-preparation
fixtures, not native upgrades; publisher trust and U2–U6 still require the actual application transition.

Installation sessions can now replace the owned package for baseline fixture setup. Replacement
is serialized, repeated requests for the same candidate share the installed result, failed
removal retains the previous owner, and failed installation leaves an absent state with the
requested package for a later explicit attempt. Cancellation waits for admitted native mutation
so cleanup ownership is not lost. Eight ownership tests cover these paths. This mechanism is
not an updater and cannot satisfy U2 by directly installing the candidate.
The session also exposes current ownership and can reset its next candidate without installing it.
This preserves lazy installation when a standalone baseline case had no preceding native install.
If baseline application cleanup or package restoration fails, the worker blocks further native
cases and records cleanup failure. It exports both baseline launch traces, process logs, package
identity, owner identities and a digest-only snapshot of persisted profile files.

Set `LAB_WORKER_BASELINE` to the previous release manifest when using
`scripts/artifact-worker-probe.ts` to exercise U1 between candidate installation and candidate CLI
verification/removal. The local macOS15 ARM64 run at
`/tmp/ml-update-baseline-verified-20260919/result.json` passed its seven selected cases with no cleanup
errors. It verified the 0.1.3 baseline through two distinct app/service instances, restored 0.1.4,
verified that candidate's bundled CLI and removed the installation. Baseline traces/logs were exported
to content-addressed evidence. Both packages came from the same source with fixture versions; this
does not qualify historical migration, native updater replacement, or Windows/Linux execution.
Thirty-one focused worker, baseline, pair and ownership tests pass, as does targeted typechecking.

Backend verification groundwork now exists in the native executor: it retains target-model
allocation locations before physical-memory aggregation, excluding draft/projector allocations and
buffers without model bytes. A successful worker completion emits bounded structured diagnostics
with model ID, worker PID/generation, private request ID and those allocations under its trace.
This does not yet pass E6: the worker still needs the integrated runtime-module identity
verification and live CPU/Metal/CUDA qualification. No inference benchmark is introduced.

`executionTelemetry()` provides a scoped private-loopback OTLP/JSON collector with bounded
retention, reviewed fields, retry deduplication and conflicting-record rejection.
`observeGeneration()` gives an ordinary endpoint generation a unique trace and correlates it with
one matching native completion, preserving public/native request IDs separately. It tolerates
delayed export without repeating inference. This adapter is not yet wired into the worker's E6
verdict because runtime-module/device verification and packaged live qualification remain required.

Native completion can now include already-loaded backend module filenames, lengths and backing-file
SHA-256 digests. The producer uses full-path loader lookup without loading absent libraries, holds
the extra reference while streaming the immutable file, and omits unavailable observations. It only
hashes when export is configured. The collector preserves missing observations as unknown and rejects
malformed supplied records. These digests must still be matched to admitted runtime artifacts in E6.

The actual Rust exporter transport passed at `/tmp/ml-native-telemetry-20260919/observations.json`
using `LAB_TELEMETRY_PROBE_ROOT=/absolute/fresh/path bun packages/testing-lab/scripts/execution-telemetry-probe.ts`.
That probe runs an explicitly ignored Rust transport fixture against the owned collector and flushes
the real exporter. Its completion/allocation values are synthetic; it is not a Metal generation test.

`scripts/native-generation-probe.ts` exercises real UI model loading and public endpoint generation
against an explicitly supplied native installation. Set `LAB_PROBE_ROOT` (an isolated profile with
the selected model already cached), `LAB_PROBE_EXECUTABLE`, `LAB_PROBE_ICN_INSTALLATION`, and
`LAB_PROBE_MODEL_ID`; `LAB_PROBE_PORT` defaults to 11339. Run with the pinned lab Bun runtime.
It owns its telemetry listener and desktop-control directory, records both correlated and collected
native evidence, and reports cleanup failures. It does not qualify the package's inference acquisition.

The real local run at `/tmp/ml-native-generation-export-20260919/native-generation.json` passed:
Qwen3.5 4B returned `HELLO`; the same trace identified worker46546, public completion
`chatcmpl-icn-1`, private request4, and 2,904,582,144 target-model bytes on native backend `MTL`
plus 521,472,000 host bytes. The physical device ID remained unknown. This run found and fixed
disabled worker export and stripped endpoint/log settings in worker launch. Planning-worker export
remains suppressed. App cleanup passed; this is real completion/allocation correlation, not a full
E6 verdict or loaded-module proof. The lab's 14 collector/correlation/endpoint tests, targeted
typecheck, native launcher regression, and actual dynamic Metal build pass.

The subsequent `/tmp/ml-native-modules-20260919/native-generation.json` run additionally verified
loaded-module observations: worker47315 generated `HELLO` and reported `libggml-cpu-apple_m4.so`
(918,120 bytes) and `libggml-metal.so` (2,060,592 bytes). Independent streamed hashing matched both
reported digests. The native loader regression proves that an existing unloaded library stays
unloaded, another directory's identical basename does not qualify, and observation leaves the actual
owner's library usable. This regression ran on macOS; Linux/Windows runtime qualification remains
outstanding. Five collector/correlation tests and lab typechecking pass. Cleanup was empty. The
explicit development installation remains a diagnostic input, not admitted release-artifact proof.

`runtimeRelease(release, host)` prepares verified private copies of admitted native base/backend
archives and serves the exact manifest through production release URL conventions on a scoped
random loopback route. It supports parallel byte ranges, rejects unknown routes, and closes before
its temporary copies are removed. Production `acquireRelease` and `downloadArtifact` pass against
the fixture, including segmented download with fallback disabled. Tests also reject missing host
bases, reserved names, wrong lengths and corrupt CAS bytes, and prove later CAS mutation cannot
change published copies. Candidate desktop and bundled CLI sessions now use this origin when the
admitted manifest includes native archives. U1 retains the validated previous manifest and gives
its baseline session a separate scoped origin; baseline cleanup leaves the candidate origin intact.
All ordinary worker sessions remove ambient development-installation overrides (including Windows
case variants). App-only manifests retain ordinary released-runtime acquisition, and cannot prove
unpublished inference changes. Source runtime compilation remains outstanding. Delivery alone does
not qualify native archive contents or application execution.

Private acceptance routing is now selected through a compiled update configuration. Set
`MAGNITUDE_UPDATE_ACCEPTANCE_CONFIG` to a JSON file when invoking the existing release
acceptance builder. It validates and copies the configuration before modifying version inputs.
The file contains `acceptance: true`, the HTTPS update `origin`, `keyId`, publisher `publicKey`
(PEM), and optionally `artifactDelivery: { "_tag": "PrivateAcceptance", "origin": "https://your-private-origin" }`.
Windows additionally needs `windowsPublisher`. Omitting artifact delivery retains GitHub delivery.
No private key belongs in this file. The fixture must sign offers using the matching private key.

Private byte transfers admit only the compiled HTTPS origin, reject all redirects, omit installation
credentials and cookies, and retain hash/size checks and native publisher verification. Production
configuration cannot select this policy; runtime environment variables cannot enable it. A local
HTTPS fixture still needs a certificate trusted by the test process; disabling TLS verification is
not part of this mechanism.

59 release tests passed, including private-policy/configuration rejection, interrupted range
recovery and corrupt-file rejection. The Mac update source passed four valid/corrupt delivery
cases across GitHub/private policies; its native installer is mocked, so these are not U2 evidence.
One Linux source regression passed; three Windows-native tests were skipped on macOS. Both private
acceptance and normal desktop bundles compiled locally. Release typechecking passed; desktop
checking retains the existing Effect multiple-provide warning at main.ts:146 (exit 2). No real
old/new native update or U1–U6 qualification is claimed yet.

The scoped `updateFixture` now creates a loopback HTTPS server, temporary certificate and ephemeral
publisher, and writes the public acceptance configuration for the build. Keep that scope alive
while building and running the acceptance pair. Pass its `caPath` as `NODE_EXTRA_CA_CERTS` only to
the test application; the host trust store is unchanged. Publish verifies and copies the candidate
before offering it. Signed checks and download resolution validate metadata, timestamps and nonce
replay; artifact routes serve the owned bytes and support ranges. Withdraw removes the offer, and
scope cleanup closes the listener and deletes certificates, private TLS key and copied packages.

Real HTTPS tests use Electron's Node runtime: the certificate fails without explicit process trust
and succeeds with it. They verify signed offers, replay/expired-request rejection, target/version
filtering, immutable delivery after source mutation, ranges, failed-publication retention, credential
rejection and listener/file cleanup. Together with pair/ownership/UI checks, 14 targeted tests pass;
testing-lab typechecking passes. The fixture is not yet wired into scheduled U1–U6 execution.
An arbitrary previously built production package cannot consume this ephemeral trust. Both private
acceptance packages must be built for the same fixture; testing an actual production baseline
requires its production-trusted release path and remains a separate qualification.

## Installed Mac private-update probe

`LAB_UPDATE_PROBE_ROOT=/absolute/fresh/path LAB_UPDATE_PROBE_TARGET=macos-15-arm64-metal-apple-silicon bun packages/testing-lab/scripts/update-probe.ts`
uses the pinned Bun runtime, snapshots unpublished source into a separate directory, installs frozen
dependencies and builds 0.1.3/0.1.4 acceptance packages for one scoped HTTPS fixture. These are fixture
versions of the same source, not a historical release migration. It installs the older DMG, changes
an appearance setting and disables automatic downloads through the UI. It first offers a same-length
corrupt copy of the newer ZIP with authentic signed metadata, requires download rejection and checks
that the old version and theme remain. It then republishes the intact archive and uses Settings to
check, download, verify and discard it. It quits and removes the owned installation.
It records build logs, source digest, host observation, Playwright trace, screenshot and cleanup errors.

The native macOS15 ARM64 probe passed at `/tmp/ml-native-update-controls-20260919/update-report.json`
with no cleanup errors. Both package builds passed. The installation, temporary trust/listener and
probe disk mounts were gone afterward. This establishes native acquisition, not U2 replacement,
relaunch, retained-data migration, post-update generation, or production publisher trust.
Local code-signing inventory currently reports zero valid identities; the fixture builds are ad hoc.

The corrupt-download and recovery probe passed at
`/tmp/ml-native-update-corruption-20260919/update-report.json`, with both builds successful and no
cleanup errors. Screenshots capture rejection and the subsequent intact download reaching Ready.
This qualifies the corrupt-download portion of U5 on this local Mac; publisher rejection, scheduled
suite integration and the rest of native update acceptance remain outstanding. The fixture test
separately proves the corrupt response preserves length and authentic metadata, changes only its
private copy, and returns exact bytes again after republishing.

`DesktopDriver.restartForUpdate()` saves the trace, invokes the normal restart control and requires
the exact retiring Electron process to exit successfully. Real Electron tests cover clean exit,
nonzero exit and an interrupted wait that leaves the application usable. This is a handoff witness,
not proof of replacement or automatic relaunch. An ephemeral self-signed identity diagnostic failed
local code-signing trust; its temporary keychain was deleted and the user search list was unchanged.
No host trust changes were made to manufacture native updater acceptance.

The probe found and fixed two issues: DMG authoring now selects HFS+ explicitly rather than depending
on the host's APFS default (the fresh-image layout regression passes), and the update checkbox driver
clicks once then waits for its asynchronous rendered acknowledgement. It does not repeat the click
or require an immediate DOM toggle. The renderer regression checks delayed state and idempotent
selection while still perturbing presentation. Fourteen lab tests and three native image tests pass;
release and testing-lab targeted typechecks pass.

### Apple package signatures

P5 now verifies the installed app resource seal and each native application/runtime file with
strict, all-architecture code-signature checks. The runtime comes exclusively from the admitted
base and selected backend archives. Its receipt records each relative path, signing identifier,
signature kind and team. Development runs explicitly report `productionTrusted: false`; an ad-hoc
signature proves integrity, not publisher trust or notarization. App-only inputs without admitted
runtime archives remain blocked for complete signature coverage.

The release profile additionally requires `LAB_EXPECTED_APPLE_TEAM_ID` in the worker's configured
environment. This is a public expected publisher identity, not a signing credential. An absent or
malformed value blocks verification. Every native signature must satisfy that Developer ID team
and contain a secure timestamp; the installed app must pass stapler validation and Gatekeeper
assessment. No candidate is re-signed and no host trust settings are changed. The production path
has fixture coverage but has not been qualified with a production-signed release. Linux P5 verification remains outstanding; the Windows implementation is described below.

Real local macOS verification at `/tmp/ml-signature-worker-20260919/result.json` passed nine
selected cases, including 37 signature records covering the app, CPU base and Metal pack; cleanup
errors were empty. A separate native negative test modified signed bytes and verified rejection.

### Windows package signatures

Windows P5 inspects the admitted installer, all installed native PE files, and the admitted
runtime archives. The required application, bundled CLI/service, desktop bridge, uninstaller
and inference executable must be present in that inventory. Development receipts retain each
file's `Valid` or `NotSigned` status, signature type, and explicitly decline production trust. Hash mismatches,
untrusted signatures and malformed native responses fail verification.

The release profile requires `LAB_EXPECTED_WINDOWS_PUBLISHER` and `LAB_WINDOWS_SIGNTOOL` in the
worker configuration. Magnitude-owned code must have a valid signature from that publisher and
a timestamp, then pass `signtool verify /pa /all /tw`. Known Microsoft CRT files under the admitted runtime
library directory retain the release pipeline's explicit Microsoft publisher exception. Other vendor DLLs retain their vendor identity; release verification stays blocked until their
expected publisher or signed-container provenance is independently verified. A generic valid
signature cannot establish that policy. No certificate is installed and no file is re-signed. Authenticode acceptance does not certify
SmartScreen reputation. Production-signed package and Windows client execution remain unqualified.

`azure-windows-signature-probe.ts` exercises the exact native PowerShell inspection script against
a pinned Microsoft-signed redistributable (never executed), an unsigned compiled fixture, and changed signed bytes. It owns
an Azure lease and releases VM/NIC/disk resources after success or failure. Its report records the
actual OS and does not qualify Windows 10/11 app behavior when using a Server diagnostic image.

The native diagnostic distinguishes embedded Authenticode signatures from catalog membership,
which PowerShell can prefer when both are available ([Microsoft documentation](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/get-authenticodesignature)).
The first Server probe correctly failed its negative-test expectation: an altered OS executable
was reported as unsigned, not `HashMismatch`. The final fixture pins the Microsoft download URL and SHA256, verifies its embedded Microsoft
signature before changing a PE section byte, and records the original file identity and digest. A
catalog change cannot be relabelled as successful embedded-signature corruption detection.

The pinned-fixture probe passed on a real Azure Windows Server 2025 guest (build26100):
`/tmp/ml-azure-windows-signature-pinned-20260919/observation.json` records Valid/Authenticode,
NotSigned/None, and HashMismatch/Authenticode for the three cases. The report has no cleanup
errors. This validates native inspection and corrupted embedded-signature detection, not a
Windows 10/11 application, a full installer, third-party publisher policy, or production signing.
