# Run the testing lab

Use the same `bun lab` command from your checkout and from CI. Local changes do not need a
commit or push. The coordinator snapshots the requested input, builds it on one disposable
machine, and tests the resulting packages on a different clean machine.

**Current qualification:** complete Ubuntu runs finished locally (65/66 passed) and in GitHub
Actions (64/66 passed), with clean resource removal and verified retained evidence. Hermes file
editing remains a real failure; one Pi terminal readiness assertion is being corrected. Other
targets remain in qualification. See the concise [coverage ledger](COVERAGE.md).

## Sign in once

Use the repository's pinned Bun version, currently 1.4.2, and Azure CLI authentication:

If your global Bun differs, invoke commands as `npm exec --yes --package=bun@1.4.2 -- bun lab …`
without changing your global installation.

```sh
az login --tenant 4581d4bf-a664-4a42-a66a-c842beeec9e7
export LAB_URL=https://magnitude-lab.lemonpebble-25e896b9.westus2.azurecontainerapps.io
export LAB_AUTH=entra
export LAB_ENTRA_TENANT=4581d4bf-a664-4a42-a66a-c842beeec9e7
export LAB_ENTRA_APPLICATION=b47912ec-a3bb-49f4-ac37-25e06b4f7743
```

The client renews identity tokens during long runs. The coordinator owns cloud credentials;
your candidate application does not receive those credentials. Resources belong to Azure
subscription `5304c4b3-d605-4193-b0cb-766c065acfa6`, resource group `magnitude-ci`.

## Test unpublished changes

Run the smaller Ubuntu app/generation/CLI selection:

```sh
bun lab run --source . --target ubuntu-24.04-x64-cpu-intel \
  --suite install,endpoint,cli --harness pi \
  --mode verify --concurrency 1 --deadline 180 --budget 25 \
  --json /tmp/lab-results.json --junit /tmp/lab-junit.xml
```

Run all nine suites and all three harnesses on that same target:

```sh
bun lab run --source . --target ubuntu-24.04-x64-cpu-intel \
  --suite package,install,app,endpoint,harness,recovery,cli,update,uninstall \
  --harness pi,opencode,hermes \
  --mode verify --concurrency 1 --deadline 180 --budget 25 \
  --json /tmp/lab-results.json --junit /tmp/lab-junit.xml
```

Dependencies are added automatically: asking for endpoint generation also installs the app,
downloads and loads the model through its UI, and verifies the requested backend. Budget is an
admission limit, not a claim of exact billing. Deadline is in minutes. No performance benchmark
is required for a passing functional result.

| Suite | What must actually work |
|---|---|
| `package` | Compile input, produce packages, verify versions, dependencies and applicable trust policy |
| `install` | Verify package bytes, use the native installer, launch, find the bundled CLI, reject corruption |
| `app` | Setup, catalog, download/load, settings, Connections, window lifecycle and error presentation |
| `endpoint` | Real generation, streaming, tool follow-up, invalid requests and backend/device attestation |
| `harness` | Pi/OpenCode/Hermes generation, session continuation, tools, resume and terminal interruption |
| `recovery` | Cancel/retry, interrupted download, offline cached generation, worker fault and reload/restart |
| `cli` | Bundled CLI version, commands, connections, interruption and independence from developer runtimes |
| `update` | Native replacement/relaunch, retained state and generation, corruption and interruption recovery |
| `uninstall` | Remove package/processes, make login registration non-runnable, retain data and reinstall |

UI tests use stable automation identities and product state, not button copy, color or placement.
Real user flows and native package behavior remain the assertions; a mocked model cannot pass
generation. Failure screenshots, traces, native logs and per-case receipts are retained.

## Reuse an existing package

Replace `--source .` with `--artifacts /absolute/path/release-manifest.json`. Artifact files must
exist beside the manifest. Their bytes are verified and uploaded; this does not rebuild them.

Source update testing builds a private same-source version pair and uses the real updater. This
does not establish historical release migration or production signing trust. `--update-from`
admits an explicit previous package graph, but its complete historical updater journey remains
unfinished. Do not treat that option as verified release compatibility.

## Reconnect, inspect or cancel

```sh
bun lab status --run run-<uuid>
bun lab wait --run run-<uuid> --json /tmp/lab-results.json --junit /tmp/lab-junit.xml
bun lab results --run run-<uuid>
bun lab evidence --run run-<uuid> --digest <sha256-from-results> --output /tmp/ui-trace.zip
bun lab cancel --run run-<uuid>
```

`--no-wait` submits and returns the run identity. Reconnect with `wait`; do not submit again just
because your terminal disconnected. Interrupting the local waiting process does **not** cancel
remote work. Use `cancel`, then `status` to verify cleanup. Evidence downloads verify their digest
and require a destination that does not already exist.

Exit zero means every selected check passed **and** cleanup completed. A blocked check is not a
pass. A failed prerequisite blocks its dependents while independent checks continue.

## CI and broader coverage

[The GitHub workflow](../../.github/workflows/testing-lab.yml) submits the same complete command,
using GitHub OIDC instead of a developer login. Blacksmith only submits and observes; Azure builds
and tests. The workflow saves reports/evidence and requests cleanup on failure or cancellation.
Its complete nine-suite run finished with two failed cases and clean cleanup; the earlier smaller run passed.

`quick`, `pr`, `full` and `release` select coverage; they are not different test engines. `--suite`
with explicit targets selects a custom subset. `release` requires trusted final artifacts and
production trust checks. Listing a target with `bun lab targets` does not mean its machine image,
driver or entire suite is qualified. Avoid `full` until the remaining provider setup is complete.

`verify` uses clean workers. `iterate` is accepted by the protocol, but warm worker/cache reuse is
not implemented yet. Use `verify` for dependable current behavior. The shared office Spark always
requires explicit `--allow-spark`; leave it out of normal development runs.
