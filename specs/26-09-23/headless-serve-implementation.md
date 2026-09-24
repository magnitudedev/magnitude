# Headless serve: implementation and verification plan

Status: implementation plan; no product changes yet. Branch: `headless`, baseline `772cfacb`.
Prepared 2026-09-23. This is the execution plan for the whole feature, including updates,
installation scripts, migration, documentation and release verification.

Inputs: the supplied `headless-serve.md`, the user's subsequent startup-update decisions, the
[updater investigation](application-updates.md), and the
[Windows updater defect](../../bugs/26-09-23/windows-update-directory.md).
This document consolidates those inputs; where they differ, the behavior below is the target.

## Product contract

1. One service owner per configured user profile: Desktop or Headless, using the existing kernel
   application lock. ACN remains an owned child with native parent-loss protection.
2. `magnitude serve` runs in the foreground without Electron. It owns ACN until normal shutdown,
   unrecoverable service failure, or cooperative desktop takeover. It does not install a boot service.
3. Desktop has priority: request Headless `Yield`, wait for full child retirement and lock release,
   then acquire ownership. Never kill a foreign owner or infer kill authority from PID/port/name.
4. `serve` refuses an existing owner or occupied port immediately after bounded observation, exits 1,
   and never takes over. A successful yield exits 0 and does not reclaim ownership.
5. Ordinary service-backed CLI commands only connect. With no service they exit 1 with:
   `No Magnitude service is running. Open the Magnitude desktop app or run \`magnitude serve\`.`
   Protocol mismatch and other errors remain distinct. `app open` explicitly launches Desktop.
6. `status` passively reports owner and service state; Desktop-only fields appear only for Desktop.
   No owner prints Not running plus startup guidance and exits 0. Failed observations are not None.
7. Remove the public `service` namespace. Login startup remains a desktop preference. Retain
   `native-runtime-check` and required private installer entries. Remove SDK `cliLayer` and its
   command-specific errors; retain the starter service where other legitimate compositions need it.
   Pi becomes connect-only; no broader Pi feature work.
8. Every installer installs the full desktop/CLI/ACN bundle and never automatically starts Desktop.
   ICN and model acquisition remain outside the bundle, under existing exact-version contracts.
9. Updates may check/download while serving, but never automatically stop a live server. Ready output
   explains: stop the server and run `magnitude serve` again. Shutdown itself does not install.
10. Next owner startup applies an already-prepared unattempted update before starting ACN, then
    continues the same foreground/headless or desktop visibility intent. No prepared update means
    no blocking update-network check. Failed attempts need explicit retry. Missing authorization
    defers and reports guidance while the intact old installation may serve.
11. Shared update commands address the existing owner without launching Desktop. Headless `install`
    refuses while serving and explains stop/start; Desktop `install` retains explicit restart consent.
    With no owner, `check` is a finite check, `download` waits for durable preparation, `install` is a
    finite maintenance transaction and leaves the service stopped, `discard` is finite, and `status`
    observes persisted state without creating an owner. No-owner mutations acquire maintenance
    ownership and recheck races; they never take a lock over an active owner or duplicate its worker.
12. Public `serve` has the agreed fixed profile/endpoint, no new remote-bind/data-dir/port flags.
    Existing development/acceptance overrides remain internal. Preserve any existing network-access
    configuration; this work introduces no new external listener or authentication behavior. Remote
    testing uses SSH or a loopback tunnel, not a new public bind.

## Architecture to implement

- **Owned application** in daemon-management: profile/resource resolution, owner acquisition,
  installation admission, legacy retirement, service command, port preflight, service supervisor,
  and control lifetime. Compose caller-owned Desktop or Headless dispatch/presentation. Do not put
  windows, login UI, terminal printing or update timers into ACN/SDK.
- **Owner contract** in SDK: `owner = Desktop { tray } | Headless {}`, plus `Yield`. Migrate every
  reader/writer together; no legacy-shaped snapshot compatibility. Because installed copies ship
  together, schema migration is atomic within a release; malformed/old peers fail clearly.
- **Shared updater** in daemon-management: Effect preparation engine, schedule, preferences,
  identity, durable store and startup reconciliation. Release owns signed protocol and channel policy.
  Client-common owns client presentation/operations; the SDK holds portable contracts only.
- **Platform installation transaction**: custom verified macOS bundle exchange; existing Windows
  NSIS transaction; existing Linux package transaction with headless authorization entry. Keep native
  filesystem identity/ACL/exec/job operations small and explicit; use Effect for TS orchestration.
- **Startup continuation**: a separate capability from installation. It preserves invocation,
  cancellation, supervision and exit status. Helpers cannot choose to open a desktop for `serve`.
- **Admission**: retain the existing per-user application lock and helper installation lease; Linux
  also needs its system-wide installation admission. Decide macOS installation-wide protection at
  Phase 2 before mutation code. Never unlink lock files, confuse file existence with lock ownership,
  or wait for an installer while retaining a lock it needs.

Before editing each area, read its AGENTS and every `bun design-docs <paths>` match. Update durable
design contracts with the implementing phase, including applicability when ownership moves. Add
schemas for persisted/IPC state, exact optional Options, branded identities and Effect DI services.

## Verified test infrastructure and preparation gaps

These are observations made during planning, not feature acceptance results.

| Environment | Verified access | Intended coverage / limits |
| --- | --- | --- |
| Local Mac | Workspace and installed Desktop available | macOS native ownership, foreground PTY, GUI takeover, signed bundle replacement, install-script shell registration; use disposable profiles/bundles |
| Parallels Windows 11 | Running VM `{976fa17d-dbd4-4576-9efd-18b4ffa7e5ba}`; `prlctl exec` works; `--current-user` is `trg` | Real Windows ACL/jobs/installer/console and GUI. ARM64 guest running supported x64 product under emulation; does not replace native x64 CI |
| Lima `magnitude-ci` | VZ Ubuntu 22.04.5 aarch64; SSH works; user systemd running; passwordless sudo available | DEB, Linux native lifetime/admission, systemd, headless and Xvfb desktop takeover. Port 10100 is already occupied: do not reuse without identifying existing owner |
| DGX `tom@sparky` | Ubuntu 24.04.4 aarch64; NVIDIA GB10; user systemd running; SSH works | Real remote headless GPU inference, restart/parent-loss, CLI over SSH; no destructive package recovery tests |
| Existing CI | Windows x64 and Linux x64 ownership jobs, signed macOS/update workflows found | Run branch-specific native jobs, both package families and supported architectures; runner/credential availability must be confirmed |

Useful entry points:

```sh
prlctl exec 'Windows 11' --current-user powershell.exe -NoProfile -EncodedCommand <payload>
/Users/trg/magnitude/specs/26-09-10/linux-vm/toolchain/bin/limactl list
ssh -F /Users/trg/.lima/magnitude-ci/ssh.config lima-magnitude-ci
ssh tom@sparky
```

- Plain Parallels exec runs as SYSTEM. That is useful for fixture provisioning, never proof of per-user
  update behavior. Use encoded PowerShell or uploaded scripts to avoid shell-quoting corruption.
- Windows toolchains exist under `C:\Users\trg\MagnitudeTesting\bun-windows-x64`,
  `node-v24.21.0`, and prior scripts reference `C:\MagnitudeBuildTools\VS`. Verify versions and actual
  availability rather than using ambient PATH. Match repository Bun 1.4.2 for acceptance.
- Lima's repository mount is read-only. Build in a guest-local checkout. Its existing scratch mount
  references an old temporary path; create a new controlled artifact exchange instead. Bun/compiler
  setup is not yet verified. Add a Fedora/RPM disposable VM or native runner: `rpm` on Ubuntu is not
  a `dnf` package-lifecycle test. Clone/prepare disposable guests before destructive tests.
- Only one Windows VM is currently listed. Parallel Windows lanes require isolated clones or CI;
  never run competing installer tests in one user/installation. Normal-user GUI and noninteractive
  runs are different lanes. SSH server availability is not yet established; a noninteractive task
  or service is an alternative for that admission test, but real SSH remains an acceptance case.
- Sparky has `~/.bun/bin/bun`, 23 GB of cached models, and an existing `~/magnitude` checkout with a
  modified inference submodule. Use a separate checkout and profile. Do not reset it or overwrite
  its models. Sudo availability was not confirmed. Run unprivileged packaged-layout acceptance
  there; exercise package installation on disposable Linux VMs.
- Signing/notarization credentials are not yet validated. Inspect presence without printing secrets.
  Ad-hoc signing can prove mechanics, not production publisher or Gatekeeper acceptance. macOS x64
  and native Windows x64 need appropriate runners if unavailable locally; record missing coverage.

## Execution and checkpoint discipline

Checkpoint commits cover substantial integrated work after its validation gates pass. Component
steps and intermediate tests stay uncommitted until the larger working chunk is verified; phases may
be combined or adapted as evidence warrants. No automatic amend/squash of accepted checkpoints.
Preserve unrelated user work.

At each checkpoint record: source commit, test commands and exit codes, OS/architecture/runtime,
artifact SHA-256/version/source commit, scenario results, skipped tests with reason, logs and remaining
limitations. Keep raw artifacts in unique ignored run directories; maintain a concise tracked ledger
beside this plan. Generated binaries/model data and credentials never enter commits.

`specs/` and `bugs/` are intentionally ignored. Explicitly add only these approved Markdown plans
and the checkpoint ledger with `git add -f`; do not unignore or add the whole trees.

Red tests must be resolved or reproduced unchanged on the baseline with their impact documented.
A related baseline failure is still a gate to fix; an unrelated failure is not grounds to claim a
full-suite pass. A platform skip, source review, cross-build or mocked ACL test never counts as
native acceptance. Retry only after diagnosis; retain the failing evidence.

Independent OS test jobs may run concurrently. Keep dependencies, commits, installers targeting one
installation, and stateful profile operations sequential. Add failure-injection controls only to
test/acceptance builds; do not expose trust overrides or destructive hooks in production binaries.

The current request is planning. Production publication, release promotion, fleet updates and broad
machine reconfiguration are separate actions; the implementation can build and test disposable
artifacts without publishing a customer release. External acceptance workflows that upload/promote
artifacts must be inspected and scoped to dedicated test infrastructure before execution.

## Phase 0 — Baseline, fixtures and evidence (checkpoint 0)

**Implement:** provision guest-local checkouts and disposable users/VMs; capture baseline status.
Create an Effect-based acceptance driver under `packages/release/scripts/acceptance/` with a scenario
registry, deadlines, evidence schemas and exact-process cleanup. Reuse existing native tests and
build helpers rather than another release pipeline. Add two-version/three-version cohort generation
using isolated worktrees because existing acceptance builders temporarily edit version inputs.
Use build-time test publisher/origin configuration; production must reject it.

**Verify:** run relevant existing CLI, SDK, daemon-management and desktop updater tests; generate build
identity before importing generated modules; targeted package typechecks. Run current native ownership
smokes on Mac/Windows/Linux. Reproduce the Windows download-directory failure with real native ACLs
under a normal user. Capture existing listener/process identities before tests; cleanup touches only
owned fixtures. Check that test configuration cannot point at production update origin.

**Gate:** working per-user Windows execution, Linux toolchain, artifact transfer, baseline report and
fresh/poisoned profile reproduction. Known release failure is recorded, not papered over.
**Commit:** `Establish headless lifecycle acceptance harness`.

## Phase 1 — Repair Windows update preparation (checkpoint 1)

**Implement:** separate scratch download creation from private prepared storage, or create the root
through its native private capability before any transfer. Add narrow recovery for the bad directory
already left by old clients, following the bug document; do not weaken the general ACL validator.
Correct packaged acceptance paths for canonical preferences/identity and preserve diagnostic causes.
Use exact private creation for every trusted artifact/helper. Preserve signed release and Authenticode
verification across retries.

**Verify:** normal inherited parent + absent updates directory reaches Ready through actual download;
existing bad directory recovers; already-private directory remains usable; cancellation/retry; wrong
owner, junction/reparse, unknown contents and interruption during recovery preserve unrelated data.
Run real native ACL tests and a download-to-stage integration test, not the current mocked stage-only
test. Build fixed A and newer B; manually install A over an affected profile, update A→B through the
app, then B→C. Check installed CLI/service version, profile preservation and process retirement.

**Gate:** native Windows update works for fresh and previously affected profiles, including the second
update. Packaged signed test certificates prove the configured test publisher only; production signer
validation is retained for final acceptance.
**Commit:** `Restore Windows application update preparation` (independently releasable).

## Phase 2 — Resolve native installation/startup risks (checkpoint 2)

Do this before building the public feature around unproven continuation behavior.

**Implement/prototype:** a minimal signed macOS helper performing staged exchange/recovery on fixture
bundles; Unix exec/adoption of a narrow installation capability; Windows copied helper plus foreground
waiting process around actual NSIS replacement. Establish lock order and installation-wide admission.
Use retained native identities, never PID files or process-name killing.

**Verify macOS:** same-volume exchange, downloaded archive on another volume, nested signatures,
framework links, Gatekeeper, unsupported filesystem and permission denial. Kill after intent/exchange/
commit and verify correct recovery. Prove exec preserves foreground tracking, cancellation and locks.
Check multi-user admission: either implement shared installation admission outside the replaced bundle
for every new owner or restrict unsupported shared-install cases with a clear deferral; a per-user
lock alone is not accepted proof. Account for prior-version owners during migration.

**Verify Windows:** loaded original CLI and addon, cwd outside the target, released owner handles,
real installer completion, deferred cleanup and next launch. Cancel/terminate the foreground parent
during installation and after the new owner starts; no orphan ACN and no detached terminal server.
Test A→B→C while prior images may remain mapped. If it works, use the minimal waiting-parent design.
If Windows refuses replacement, implement an external stable launcher in a separate sub-checkpoint,
including PATH, exact child launch, console/job lifetime and launcher-update policy, then rerun the
same tests. Do not silently settle for exit-and-hope or require another manual command after startup.

**Windows decision (native probe, 2026-09-23):** A→B succeeds while the original compiled runtime
and addon remain mapped; B→C returns installer exit 1 and stays at B. Releasing the original process
allows C to install successfully. The fallback is therefore selected: a small foreground launcher
outside the application payload, with a separate minimal protocol and explicit maintenance policy.
The waiting-original-CLI approach does not meet repeated-update acceptance. The probe uses the
production NSIS transaction with an inert application payload; it is not signed-runtime or inference
acceptance. The native launcher prototype subsequently passed two installed continuations,
argument/cwd preservation, parent-loss cleanup and console cancellation. Release integration remains
gated on launcher maintenance and preserving independent desktop launch from finite CLI commands.
Uninstall ordering now retires a retained previous payload or defers while registration remains
available; native acceptance covers refusal, retry, fresh reinstall and interrupted replacement.
Finite command lifetime is separated from serving containment; real desktop and packaging tests remain.

**Verify Linux:** installation helper vs application/shared installation locks; authorized and denied
sudo flows with the actual deb transaction; helper lifetime under systemd cgroup cleanup. Preserve the
invoking user for resumed service. No background password prompt and no root inference server.

**Gate:** record the chosen continuation mechanism per OS and its executed evidence. If infrastructure
blocks proof, finish independent phases but do not implement dependent claims as established facts.
**Commit:** `Add verified application installation handoffs` (or separate OS commits if large).

## Phase 3 — Extract shared application ownership (checkpoint 3)

**Implement:** move profile/resources/service-command construction and common bootstrap from desktop
into daemon-management. Preserve production/dev/acceptance paths, embedded Windows native bootstrap,
port checks, legacy standalone retirement and service supervisor. Add explicit output mode to owned
child command: diagnostic-tail only or tail plus parent stderr. Desktop stays tail-only. Keep native
Windows private pipes/jobs and Unix watchdog behavior. Do not rename unrelated product state.

**Verify:** existing desktop lifecycle and updater tests unchanged; command resolution from symlinked
Mac CLI and installed Linux/Windows resources; development source ACN and local ICN path; output-tail
bound under heavy stdout/stderr; sink failure cannot bypass retirement; child crash/restart and cleanup
failure. Native parent death removes descendants on all three OSes. Desktop GUI/tray/login still work.

**Gate:** behavior-preserving extraction with native smoke evidence and targeted typechecks.
**Commit:** `Share owned application bootstrap`.

## Phase 4 — Two owners, foreground serve and takeover (checkpoint 4)

**Implement:** SDK owner union and `Yield`; update Desktop snapshot consumers together. Add lazy
`serve.ts`/`serve-runtime.ts`. Acquire ownership, exclude active installation, acquire Linux shared
installation admission via native close-on-exec lease, retire previous standalone installs, perform
one-time port preflight, then spawn ACN and serve control. Keep per-attempt port admission as well.
Resolve service/addon as siblings of the real packaged executable. Headless login actions fail with
desktop guidance. Interim headless update actions must not launch Desktop before Phase 7.

Desktop acquisition observes contention: forward to Desktop; yield Headless and reacquire within a
single deadline. Derive the bound from actual child-retirement budgets with margin (60 seconds is a
starting upper bound to validate), and handle racing desktops/headless starters. A second signal does
not skip retirement. Failed/CleanupFailed headless service states terminate with failure and evidence;
they never sit forever waiting for Retry. Yield acknowledgement is delivered before endpoint teardown.

**Verify:** table-driven request dispatch and schema round trips; existing Desktop/Headless/unresponsive
owner; missing addon; busy port; active installer; failed admission; legacy retirement failure; no spawn
on refusal. Exercise cold socket retry and control reply delivery. Test graceful SIGINT/SIGTERM/Windows
console control, repeated signals, forced owner death, child crash recovery and terminal failure.
Race two desktops, many `serve` attempts and a `serve` arriving during yield; assert at most one ACN
over the full event timeline, not only at final observation. Unresponsive owner times out unmodified.

**Gate:** real packaged `serve` on Mac, normal-user Windows and display-free Lima; native takeover on
Mac/Windows and Xvfb Linux; no process leaks, no inherited installation lock in ACN.
**Commit:** `Add foreground serve with cooperative desktop takeover`.

## Phase 5 — CLI cutover and connect-only clients (checkpoint 5)

**Implement:** add passive `status`; adapt `app open` to launch Desktop when Headless is observed;
remove service commands and implicit desktop starters from inference/connections. Remove SDK CLI
starter/errors and Pi use. Preserve hidden `native-runtime-check`. Update CLI docs/help, affected
fixtures and CI commands in this same phase so the checkpoint builds and tests coherently.

**Verify:** subprocess help/version/invalid-syntax/lazy-import tests with absent native installation;
status absent/starting/ready/failed and owner-specific fields; unavailable active-model observation
prints Unavailable, not None. Every service-backed command fails with the exact no-service message
without spawning Desktop/ACN. Protocol mismatch remains an error. `app open` against Headless causes
one cooperative takeover; against Desktop focuses the existing app. Pi typecheck/build proves no
deleted starter references. Scan executable code/workflows/docs for obsolete public commands.

**Gate:** targeted CLI/SDK/Pi/client-common/desktop checks; native passive-command no-mutation evidence.
**Commit:** `Make CLI clients connect only and replace service commands`.

## Phase 6 — Shared updater and macOS installer (checkpoint 6)

**Implement:** extract preparation/schedule/identity/preferences into shared Effect services. Preserve
one timer per owner, automatic-download cancellation, channel policy, verified persistence, explicit
retry and passive status. Implement the custom macOS transaction from `application-updates.md` using
the Phase 2 proven primitives. Replace Electron updater installation wiring and its loopback staging
server; retain bounded observation of a still-active old updater job for migration safety. Update native
signing/build inputs. Generalize installer outcomes away from `showWindow`/desktop-only relaunch.

**Verify:** existing engine tests follow the move, plus clock-driven schedule tests, manual check
deduplication, preference persistence, cancellation cleanup, bad/tampered archives and interrupted
attempt records. Native Mac full bundle exchange/recovery at each durable boundary, signature and
publisher rejection, mixed/wrong version/arch, cross-volume staging, disk-full and sync failures,
cleanup failures and no accidental reverse exchange. Test old updater coexistence without concurrent
mutation. Repeat Windows Phase 1 acceptance after extraction; Linux authorization regressions.

**Gate:** signed packaged Desktop A→B→C with CLI and ACN matching on Mac and Windows; Linux package
transaction acceptance. Test-specific publisher evidence is distinguished from production signing.
**Commit:** `Share update preparation and replace Electron macOS installation`.

## Phase 7 — Headless startup updates and CLI maintenance (checkpoint 7)

**Implement:** compose the updater in `serve` after admission; prepare/report only during service
lifetime. Reconcile prepared state before creating ACN on next startup; continue the same invocation
using Phase 2 handoff. Implement the update-command matrix in the product contract. For finite
no-owner download, cancellation cleans scratch and cannot publish partial Ready state. Linux explicit
maintenance accepts validated sudo caller identity alongside existing Polkit identity; read trusted
keys from protected installed resources, never caller-provided keys.

**Verify:** hold a real long-running inference stream while an update reaches Ready; assert original
owner/ACN identity and uninterrupted stream. Normal shutdown performs no installation. Next startup
installs before the first Booted/Start/Ready event; ready server reports the new release. Offline startup
without pending bytes is prompt. Test interrupted/failed/prepared/completed records, explicit retry,
authorization unavailable, cancellation during startup handoff and competing Desktop startup. No-owner
check/download/install/discard must never open Desktop; no mutation-reply replay. Headless install
refuses without changing the live service. Mac/Windows/Linux foreground and service-manager lanes.

**Gate:** A→B→C headless startup updates, including long-lived old executable cases on Windows, with
no detached server, duplicate owner, retry loop or corrupted profile. Linux deferred state is explicit.
**Commit:** `Apply prepared updates before headless startup`.

## Phase 8 — Full-install scripts and command registration (checkpoint 8)

**Implement:** `install.sh` for macOS/Linux and `install.ps1` for Windows, through release-owned
metadata/channel resolution. Verify signed manifest binding and artifact digest; native publisher
verification before execution. Linux uses apt/dnf packages; Windows uses signed `/S` installer;
macOS invokes the extracted bundle's verified CLI installer and the same transaction used for updates.
Factor macOS CLI registration/PATH constants into shared host code. Honor ZDOTDIR, bash profile
precedence and fish; preserve user-edited blocks and conflicting commands. Windows updates the current
shell PATH after installation. No installer launches Desktop or registers headless boot service.

**Verify:** fresh/install-over-existing/running-owner/occupied-install-lease; repeat install; bad
manifest/signature/hash/arch; interrupted downloads/replacement; spaces/non-ASCII paths; offline
failure before mutation. Command works in a new shell after install. Mac symlink resolves the installed
bundle and marked shell blocks match shared constants exactly; test already-edited/missing/read-only
shell files. Native DEB and RPM dependency/install/upgrade/refusal/abort-repair/uninstall tests in
their actual OS families. Windows current-shell PATH and uninstall retain unrelated entries.

**Gate:** script install → `serve` → CLI query works without ever launching Desktop on each OS.
**Commit:** `Install complete Magnitude applications from shell scripts`.
Publishing script URLs/website changes remain a separate release action; local/acceptance hosting is
sufficient for implementation tests.

## Phase 9 — Packaged remote and service-manager acceptance (checkpoint 9)

**Implement:** consolidate reusable end-to-end scenarios into maintained acceptance scripts and CI.
Provision test-only systemd/launchd/Windows service-manager examples; do not add a public service
installation command. Fix phase defects in focused commits with their regression tests.

**Sparky runbook:** transfer the exact Linux arm64 packaged payload and manifests into a dedicated
run directory; run without DISPLAY/WAYLAND and without relying on developer PATH. Use the internal
isolated acceptance profile and an available loopback port. Prove health, hardware detection, catalog,
model acquisition/cache reuse, compatible CUDA ICN loading, generation/stream cancellation and model
unload/reload. Use a known compatible existing model or a bounded test model selected from authoritative
catalog data. Cache downloads separately; do not mutate existing model files. Record backend/driver,
engine identity, output and process tree. Exercise Ctrl+C, SIGTERM, SSH loss for a foreground session,
and systemd restart separately. A systemd-owned server must survive SSH disconnect; a foreground
SSH-owned server must not leave an unowned tree. Tunnel loopback if a Mac client is needed.

**Cross-platform final cases:** direct terminal start/stop; native Desktop takeover; fresh boot-style
noninteractive start; restart on genuine failure; successful yield must not cause restart storms
(use `Restart=on-failure`, not unconditional restart, in examples; explain contention failure policy).
Test update-ready notification, scheduled/explicit user restart, failed-update recovery, repeated
updates, package refusal while serving and preserved session/config/model state. Reuse A/B/C artifacts
from the final code; an earlier phase's binary is not final acceptance evidence.

**Gate:** scenario receipts on all available local machines plus native x64 CI; real inference on Mac
and Sparky and a small CPU model on Windows/Linux. Missing signing, architecture or RPM coverage is
reported as incomplete, not downgraded to a mocked substitute.
**Commit:** `Verify packaged headless lifecycle and startup updates across platforms`.

## Phase 10 — Documentation, release readiness and final checkpoint

**Implement:** coherent user docs and durable design updates. Cover owner priority, connect-only
errors, foreground logs/shutdown, status, install scripts, update command semantics, deferred Linux
authorization, service-manager examples and SSH usage. Explain old Windows clients need direct
installation of a fixed build and repaired cache state. Update CI trigger paths to include relocated
updater/native code and support branch-specific acceptance; existing hosted acceptance currently
auto-runs on `app`, so it must not be assumed to run on `headless`.

**Verify:** `bun design-docs --changed --explain`, targeted typechecks and affected suites, removed-symbol
search, all public help examples, matched packaged version/protocol metadata, clean install and final
upgrade receipts. Use release-owned Changesets/protocol allocation tools; do not hand-edit generated
versions or publish to bypass missing acceptance. Review every checkpoint's deferred gates and close
them or state the specific external blocker. Validate documented commands in disposable shells.

**Gate:** no known feature-related red tests; no claimed unsupported coverage; code, docs, commands,
package assets and update policy agree. Ledger identifies the final commit and exact accepted hashes.
**Commit:** `Document headless operation and finalize release readiness`.

## Mandatory existing-product regression lane

Headless acceptance is additive. A working `serve` does not compensate for breaking ordinary desktop
use or CLI use against a desktop-owned service. These regressions are checkpoint gates, not optional
final polish. Establish baseline observations in Phase 0 and retain comparable evidence.

### Drive the actual desktop with computer use

Use the computer-use tool's native app/browser controls, screenshots and visual inspection to operate
the installed macOS app. Click/type through real user entry points, including native menus and tray;
do not substitute direct RPCs, injected JavaScript or synthetic state for the user interaction being
tested. Accessibility/DOM automation and Playwright remain useful complementary tools. Use isolated
test profiles and artifacts, preserving the user's ordinary running app and data.

On Windows, use the visible Parallels guest desktop through computer use where possible; confirm
guest focus, normal-user session and app identity before interacting. For Linux, provision a disposable
graphical guest with a visible display or screen-sharing connection when needed. Xvfb/Playwright
checks remain automated coverage, but are not a claim of visually driving an otherwise unseen native
desktop. A missing VM GUI transport is recorded as missing coverage, not a successful visual test.
Use the same computer-use entry points for screenshots; do not use the voice-only screen-context API.

| ID | Real-user scenario | Required evidence |
| --- | --- | --- |
| R1 | Launch installed Desktop normally, navigate Discover/Models, Status/Usage and Settings | Usable rendered screens, no stuck loading/error state, one Desktop owner and one ACN |
| R2 | Select/load a small compatible model through UI, perform a real inference request through an existing supported client, observe live UI state, stop/unload through UI | Model and usage displays agree with authoritative results; cancellation/unload completes |
| R3 | Close the window, reopen through native tray/menu/Dock, use `app open`, then Quit and reopen | Expected visibility/focus, no duplicate owner, complete descendant retirement on Quit |
| R4 | Change an ordinary setting and update preference; exercise desktop-only login-startup preference in a disposable user and restore it | Setting persists after relaunch; native startup registration matches UI; headless work has not removed desktop capabilities |
| R5 | Check/download/update using actual Settings or tray controls, then use the updated desktop | Visible Available/Ready/error states are accurate, restart is explicitly initiated, model/session/config state remains usable |
| R6 | Start Headless, open Desktop for takeover, then use the UI and CLI normally | Single owner transition; working model operations and visible UI after takeover, not merely a passing health endpoint |

Record screenshots at meaningful before/after states, action steps, app/artifact version, process
identities, logs and authoritative service observations. Check visual problems such as blank windows,
disabled controls, stale state and misleading status. Screenshots alone do not prove backend behavior;
backend calls alone do not prove a working UI. Review relevant renderer/main-process errors.

### CLI without a headless owner

Run a dedicated lane in which `serve` is never launched:

- Start Desktop normally, then use packaged CLI `status`, hardware/catalog/model observations and
  connection commands against its service. Exercise a supported model mutation and see the desktop
  reflect the change; perform the inverse UI mutation and confirm CLI observations update.
- Exercise CLI update check/download/status while Desktop owns the service; verify the shared state
  in Settings. Exercise explicit desktop update installation in the update cohort lane.
- Verify commands leave the same owner/service alive, open no extra window, and create no additional
  service. `app open` must focus/reopen the existing Desktop rather than duplicate it.
- Quit Desktop: passive help/version/status still work without starting anything. Service-backed
  commands give the new connect-only guidance. This absence behavior is an intentional change;
  successful CLI operation with an already-running Desktop is a required preserved behavior.
- Repeat after a packaged update and after a Desktop takeover. Preserve protocol-mismatch diagnosis
  and distinguish observation failure from an empty result.

### When this lane blocks a checkpoint

- Phase 0: baseline R1–R4 and desktop-owned CLI lane; capture existing issues separately.
- Phase 3: rerun R1–R4 and desktop-owned CLI lane after shared bootstrap extraction.
- Phases 4–5: add R6, native reopen/focus and all connect-only/desktop-owned CLI cases.
- Phases 6–7: add R5 and repeat desktop-owned CLI and UI operations after A→B→C updates.
- Phase 8: R1/R3 and desktop-owned CLI against script-installed packages.
- Phase 9/final: full R1–R6 and CLI lane on final artifacts, with actual visual operation on Mac and
  Windows, and Linux visible-GUI coverage or an explicitly outstanding coverage gap.

Each checkpoint receipt records these scenario IDs and results. UI failures caused by the change
block the checkpoint even when headless/native tests pass. Repeat the affected scenario after fixes;
do not rerun unrelated expensive scenarios without a reason.

## Test commands and evidence rules

Use package-targeted checks, not project-wide `tsc -b` or bare `bun vitest`:

```sh
bun packages/version/scripts/generate-version.ts
bun packages/daemon-management/scripts/build-native.ts
# From each affected package directory:
bunx --bun vitest run <relevant-test-files>
bun run typecheck
# CLI/Desktop configs are rooted in their own packages:
bunx --bun vitest run --config cli/vitest.config.ts
bunx --bun vitest run --config desktop/vitest.config.ts
```

Daemon-management, SDK, release, client-common and Pi each run their own targeted checks when affected;
desktop/web use their package tsconfig explicitly if no typecheck script exists. Native Windows uses
`test-windows-native.ps1`, `test-windows-installer.ps1`, and actual compiled Bun/Node interoperability.
Linux extends `linux-installed-lifecycle` and `linux-update-rollback`; run the latter only in a clean
disposable guest as its existing preflight requires. Maintain these commands in the acceptance driver
so new test filenames are discoverable and platform-gated skips are surfaced in its summary.

For each destructive transaction boundary, test both injected operation failure and process termination.
For recovery/durability claims, add disposable-VM abrupt shutdown where supported; a process kill is
not proof of power-loss durability. Concurrency checks use synchronization barriers and recorded event
order, not arbitrary sleeps. A cleanup assertion checks retained process identities plus descendants,
lock availability, endpoint/port state and relevant package state; absence of a top-level PID alone
does not prove cleanup. Screenshots supplement GUI tests but are not service-readiness evidence.

## Completion definition

The feature is complete when a user can install the full package, run headlessly without Electron,
use client commands, stop or yield cleanly, receive prepared-update guidance, and start the next time
on the updated version without losing foreground/service-manager ownership. This must hold for the
supported packaged platforms, with the documented authorization exception on Linux. The Windows
updater defect must be fixed for existing affected state, not only a fresh test profile. Every phase
has a tested commit and a reproducible receipt; no production publication is implied by completion.
Ordinary desktop operation and CLI use with a desktop-owned service must also pass the mandatory
regression lane, including real UI interaction and post-update use.
