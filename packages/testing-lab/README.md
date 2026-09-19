# Magnitude testing lab

This package is under active implementation. The coordinator protocol, persistence, source
transport, provider allocation adapters and initial functional drivers exist. A deployed
scheduler, complete case implementations, image qualification, artifact-mode CLI, and CI
workflow are still required before the full target matrix can execute.

Use the Bun version pinned by the root `packageManager` (currently 1.4.2). `bun lab help`
describes the CLI. `bun lab targets` lists the requested coverage; listing a target does not
mean its provider image or backend has been qualified.

```sh
bun lab run --source . --target macos-26-arm64-metal-apple-silicon
bun lab run --source . --profile pr --concurrency 4 --budget 150
bun lab status --run run-<uuid>
bun lab results --run run-<uuid>
bun lab cancel --run run-<uuid>
```

`LAB_URL` and `LAB_TOKEN` select an authenticated coordinator. The CLI obtains owner/trust
from `/v1/me`, snapshots dirty source and initialized submodules, queries which source objects
are missing, uploads only those objects, registers the immutable input and submits the run.
No commit or push is required. Interrupting the waiting CLI leaves the remote run running;
use `cancel` for cancellation. `--no-wait` returns after submission.

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
  It requires `LAB_PROBE_MODEL_ID` and `LAB_PROBE_MODEL_NAME`; `LAB_PROBE_CACHED=true` explicitly
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

The database integration test launches an isolated temporary PostgreSQL instance. Its current
fixture discovers the Postgres.app 17 installation on this Mac; portable test-runtime discovery
is still needed for remote CI. Provider fixture tests never claim to exercise real cloud VMs.
