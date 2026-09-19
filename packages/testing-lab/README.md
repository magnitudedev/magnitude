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
status, results and cancellation. Authentication currently supports server-mapped credentials
for private development deployments; production Entra/GitHub federation remains to be wired.
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
TUI interaction, complete suite integration, GPU/backend receipts and the full OS matrix remain
unqualified. No performance benchmark gates have been introduced.

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

`scripts/source-build-probe.ts` exercises these phases on a real local source snapshot. Set
`LAB_BUILD_PROBE_ROOT` to a new directory and `LAB_BUILD_PROBE_TARGET` to the actual local target.
The lower-level candidate script accepts `LAB_BUILD_PHASE=compile|package|all`, `LAB_BUILD_OUTPUT`,
`LAB_BUILD_SOURCE_DIGEST`, and `LAB_BUILD_SOURCE_COMMIT`.

This initial integration builds on the allocated target before installation. Separate producer and
consumer machines, shared producer deduplication, private engine payload publication, and native
Windows toolchain configuration are still required for full remote verification.

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
Version checks alone do not qualify an immutable worker image. H7 terminal interaction remains
unimplemented, and the backend gate still blocks scheduled generation acceptance.

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
both pinned image observations and an installed guest runtime. This entry point does not yet
configure Azure/Spark execution or provision fresh guest runtimes. Azure Blob storage can replace
the file storage using `{"kind":"azure","config":<AzureArtifactConfig>}`.

For remote deployment, terminate HTTPS at the ingress; the bearer credential mode is the initial
private-service path. Entra/GitHub OIDC, outward worker polling, Bicep deployment and managed
coordinator hosting remain unfinished. A real HTTP/PostgreSQL test covers upload, admission,
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
