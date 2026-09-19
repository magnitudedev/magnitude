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
scoped uninstall. Unconnected cases return blocked outcomes. The scheduler transport has
not yet been wired to this worker, and source builds still need their worker path.
`artifact-worker-probe.ts` is an explicitly limited five-case install/CLI diagnostic, not a
full-profile run; it has passed against the local macOS 15 DMG with verified removal.
