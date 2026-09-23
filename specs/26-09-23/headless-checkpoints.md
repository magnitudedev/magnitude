# Headless implementation checkpoints

Execution authority: [full plan](headless-serve-implementation.md).
Branch: `headless`. Planning baseline: `772cfacb`.

## Planning checkpoint

Completed 2026-09-23: consolidated the original feature proposal and update investigations; inspected
native/package acceptance scripts; verified read-only connectivity to local test environments.
This checkpoint changes documentation only. No implementation tests, updater reproduction, native
installation, VM reset, service stop, release upload or publication was performed in this planning turn.

Environment observations:

- Parallels `Windows 11` running; ARM64 guest; SYSTEM and `--current-user` execution verified. Current
  user is `trg`. x64 Bun and Node tool directories located; their pinned versions still need checking.
- Lima `magnitude-ci` running on VZ, Ubuntu 22.04.5 aarch64, user systemd and sudo available. Port 10100
  occupied. Repository mounted read-only. Lima binary located in `specs/26-09-10/linux-vm/toolchain/bin`.
- `tom@sparky` reachable, Ubuntu 24.04.4 aarch64 with NVIDIA GB10, user systemd running. Existing
  checkout has a modified inference submodule. Use a new checkout; no changes made remotely.
- Existing CI provides x64 OS lanes and hosted/signed acceptance infrastructure. Access to required
  credentials/runners and RPM/macOS-x64 runtime coverage remain to be verified.

## Phase ledger

Planning amendment: existing-product regression is a mandatory lane. The full plan now assigns R1–R6
real desktop scenarios and a desktop-owned CLI lane to the relevant checkpoints. Native computer use
and visual inspection on Mac/visible VM desktops complement automated API and process assertions.
This amendment defines future tests; it does not claim those UI scenarios have already been run.

| Phase | Status | Commit / receipt |
| --- | --- | --- |
| Planning | Complete | This documentation checkpoint |
| 0: baseline and harness | In progress | Local native build and baseline suites executed; desktop shell probe failure under investigation |
| 1: Windows preparation repair | Implemented; packaged acceptance pending | Windows native, staging, and transfer regression gates passed in this checkpoint |
| 2: native continuation/admission proof | In progress | Windows mapped-parent probe requires external launcher; Mac/Linux gates pending |
| 3: shared owner extraction | Pending | — |
| 4: serve and takeover | Pending | — |
| 5: CLI cutover | Pending | — |
| 6: shared updater/macOS transaction | Pending | — |
| 7: startup updates/maintenance | Pending | — |
| 8: install scripts | Pending | — |
| 9: packaged/remote acceptance | Pending | — |
| 10: documentation/release readiness | Pending | — |

## Required entry for every implementation checkpoint

- Commit identity and changed behavior; record the parent source identity used for test builds.
- Exact commands, runtime/compiler/OS/architecture and artifact hashes/versions.
- Passing scenarios; failed scenarios with diagnosis; skipped scenarios and missing infrastructure.
- Desktop R1–R6 scenarios applicable to this phase, desktop-owned CLI results, and visual interaction
  evidence; explicitly identify VM GUI coverage that was not exercised.
- Evidence directory/CI receipt and owned-resource cleanup result.
- Decisions resolved (particularly Phase 2) and remaining gates affecting later phases.
- Next phase and its entry prerequisites.

Do not mark a phase passed while its required native evidence is missing. Commit independently
complete improvements if an external gate blocks other work; clearly retain the blocked gate here.

## Baseline execution, 2026-09-23

Working source: planning checkpoint `6342a59d`; product source unchanged from `772cfacb`.

- `bun packages/version/scripts/generate-version.ts` and
  `bun packages/daemon-management/scripts/build-native.ts` passed on macOS arm64.
- From `packages/daemon-management`, `bunx --bun vitest run`: 35 files passed,
  252 tests passed, 7 platform/optional tests skipped.
- `bunx --bun vitest run --config desktop/vitest.config.ts`: 30 files passed,
  1 failed, 3 skipped; 254 tests passed, 1 failed, 5 skipped. The existing shell environment
  descendant-retirement test fails because its PID fixture file is absent. An isolated rerun
  reproduces this failure; diagnosis is pending. This is not a passing regression baseline.
- Windows 11 ARM64 guest, ordinary user, existing x64 native adapter and Bun 1.3.14:
  a disposable directory created with ordinary mkdir and mode 0700 inherited permissions;
  native private-directory preparation rejected it with the reported error. Direct native
  creation and repeated preparation succeeded with the explicit protected user-only ACL.
  The disposable reproduction directory was removed. This confirms the primitive failure,
  not the complete download/install path or acceptance of the eventual fix. Current pinned
  runtime and freshly built adapter remain required for acceptance.
- Local logs: `/tmp/magnitude-headless-native-baseline.log`,
  `/tmp/magnitude-headless-desktop-baseline.log`, `/tmp/magnitude-headless-shell-baseline.log`.
  Real desktop interaction, desktop-owned CLI, and packaged upgrade gates remain pending.

## Windows preparation checkpoint, 2026-09-23

Parent: `6342a59d`. This checkpoint implements separate transfer scratch storage, narrow native
recovery of inherited Windows update caches, actionable refusal, and regression coverage. Generic
private-file validation is unchanged. Retired caches are preserved; recovery never adopts their
prepared records or executable bytes. Hosted acceptance now uses current configuration paths and
begins with an inherited cache.

Executed verification:

- Pinned Bun 1.4.2 (`744846f84`) downloaded into isolated tool directories on Mac and Windows;
  the first baseline runs used the previously installed Bun 1.3.14. No global runtime was replaced.
- Windows 11 build 26200, ARM64 guest with x64 execution, ordinary user. Compiled the native
  security acceptance executable with MSVC `/W4 /WX /O2 /std:c11`; it passed private creation,
  inherited-cache recovery, unknown-file preservation, broad-ACL refusal, and repeat recovery.
- Built the full Windows native addon with the repository build script. Node 24.21.0 and Bun
  1.4.2 both passed the new root/child junction refusal fixture, including unrelated-file preservation.
- In an isolated Windows checkout, `bun x --bun vitest run --config desktop/vitest.config.ts
  desktop/src/windows-update-source.test.ts desktop/src/hosted-update-source.test.ts`: 5 passed.
  Staging uses real native ACL operations for fresh and inherited profiles and rejects corrupt or
  unverified installer fixtures. The transfer test substitutes HTTP transport; it does not prove
  hosted release delivery or actual publisher verification.
- Windows `prepared-update.test.ts`: 6 passed, including interrupted-transfer cleanup.
- macOS arm64, pinned Bun: daemon-management full suite 252 passed, 7 skipped; desktop suite with
  `--exclude '**/shell-env.test.ts'` 249 passed, 6 skipped. Targeted daemon-management and desktop
  typechecks passed with existing diagnostic notices. `git diff --check` passed.
- The pre-existing shell test passes alone but fails after earlier tests with a protected-command
  lifetime-channel connection failure before the fixture writes its PID file. It also fails on
  pinned Bun; the full desktop gate remains unresolved. Temporary diagnostic changes were removed.

Windows artifacts in the isolated `MagnitudeTesting/headless-03a20624-69c4-4ad9-9501-8e2236452b6a`
directory:

- `desktop-host.node` SHA256 `fb336e907f23d1a568cfc1b93de1d428d997892030c9157bfdaa45e1c2b70433`.
- `windows-security-test.exe` SHA256 `748d9061448dc2364e5f2e3fba6b35c9958934e3a25987b18dab0cdc7e36a548`.

Disposable ACL/staging/junction fixtures cleaned themselves up. The isolated checkout, toolchains,
and build artifacts are retained for subsequent phases. No installed application or user profile
was modified, and no release was published. Logs remain under `/tmp/magnitude-headless-*` on Mac.

Pending: signed hosted download and two successive installed upgrades, cancellation/interruption
and rename-race acceptance, native x64 Windows lane, desktop R1–R6/desktop-owned CLI/visual gates.
Phase 1's implementation checkpoint does not claim those broader acceptance gates have passed.
Next: foreground startup continuation/admission proof, alongside completion of baseline GUI and
process-channel regression work.

## Native continuation investigation, 2026-09-23

Parent: `f9b148d7`. User authorized replacement of the existing test installations in Windows and
Ubuntu VMs. The Windows desktop was stopped through its installed CLI and uninstalled normally;
its model/profile data was not removed. A temporary acceptance account created during isolation
setup was removed without using it for installation.

The new continuation fixture builds the production NSIS installer with a compiled Bun 1.4.2 CLI
probe and the actual x64 native addon. Under the ordinary Windows user, with the foreground process's
working directory outside the application, it records:

| Operation | Installer exit | Installed version |
| --- | --- | --- |
| Install initial A | 0 | 1.2.3 |
| A→B while A CLI and addon stay mapped | 0 | 1.2.4 |
| B→C while that original process remains mapped | 1 | 1.2.4 |
| Retry C after the original process exits | 0 | 1.2.5 |

This falsifies repeated replacement with a waiting original CLI. Phase 2 now selects the external
native foreground launcher fallback. The fixture is an observation tool, not a passing assertion
that foreground continuation is implemented. It does not contain ACN/ICN, test production signing,
or establish signal/job behavior. Those gates remain required for the actual launcher.

Evidence: `continuation/foreground-continuation.json` in the isolated Windows testing directory
recorded above. The probe's finally block released its exact child and uninstalled its inert payload;
subsequent checks confirmed the installation directory and uninstall registration were absent.
MSVC helper/probe compilation, three production NSIS packages, compiled-runtime/addon loading, and
the dedicated fixture typecheck passed. Build artifacts remain available for the next experiments.

Mac installed 0.1.5 baseline: real computer-use navigation Status→Catalog, catalog search reducing
55 entries to 5, restoring search, and returning to Status all worked. A screenshot confirmed Ready.
The bundled CLI's `service status`, `models status`, and `hardware` succeeded without `serve`.
Closing the real window retained Ready service and CLI model access; `app open` restored the same
Status page. Model acquisition/loading, full Quit, branch-build UI, and R1–R6 coverage remain pending.
This records baseline behavior of the installed app, not acceptance of changed branch binaries.

## Windows foreground launcher primitive, 2026-09-23

Parent: `f3762f0b`. Added a native foreground launcher core and a known-folder installation entry
point, with an independent native build script. It is not wired into release payloads or PATH yet.
The shared atomic Job Object primitive now supports foreground console/cwd spawning while preserving
its existing hidden-child behavior. The core retains the original command context, moves its own cwd
outside the replaceable tree, contains the compiled command, observes complete retirement, and allows
one continuation only after a changed executable identity. Cancellation prevents continuation.

Executed under the ordinary Windows user, using x64 MSVC `/W4 /WX` and Bun 1.4.2:

- Existing native job suite passed: nested containment, root-versus-tree retirement, parent death,
  selected handle inheritance, denied breakaway, and creation failure.
- New launcher native suite passed: force-killed launcher retires its child and ordinary descendant;
  console Ctrl+Break retires the foreground tree and returns cancellation status.
- Production NSIS fixture A→B and then B→C passed through the same stable launcher executable in two
  foreground invocations. Each invocation continued into the installed replacement, preserving an
  empty argument, spaces, quotes, trailing backslash, Unicode, and cwd. Spoofed LOCALAPPDATA did not
  redirect installation lookup. Closing input returned success. Unchanged-image continuation failed.
- The compiled probe loads the real native addon. Its continuation now requests explicit process exit
  after Effect teardown; an exit-code assignment alone left Bun waiting on its open input stream.
- Dedicated TypeScript fixture checking passed. These fixtures contain no ACN/ICN or signing proof.

Artifacts: `launcher-acceptance-2/` under the previously recorded Windows test directory. Successful
probe output is also in the task tool receipt. Both earlier harness errors were corrected: Windows
PowerShell output decoding needed UTF-8, and the compiled probe needed explicit continuation exit.

**Packaging remains gated.** Ordinary desktop launch cannot inherit the foreground server's kill-on-
close job: the current CLI starts the desktop as a detached child, which still inherits Windows job
membership. Finish command dispatch/desktop-launch separation, stable-launcher installation and
maintenance, PATH ownership, real desktop/CLI regressions, and installation-time cancellation before
shipping this entry point. The current native entry point is only exercised by the acceptance fixture.

The fixture also reproduced an installer defect: after an update retains a mapped previous image,
uninstall removes the current payload and registration without retiring the previous tree. The next
install then refuses recovery because registration is gone. After the successful probe, application
and uninstall registration are absent, no fixture process remains, and the stage retains fixture B
(1.2.4). Preserve this evidence until adding an uninstall-recovery regression and fixing transaction
ordering; never make ordinary installation delete an unverified retained tree.

Linux preparation: installed 0.1.5-46 desktop remains Ready, and its CLI `service status` and
`models status` work without headless mode. Added GCC/build-essential and unzip in the disposable
Ubuntu VM, and isolated Bun 1.4.2 at `/home/trg.guest/magnitude-headless.uAol17lC/bun-linux-aarch64/bun`.
Sparky now has an active user service, unlike the earlier inventory; use a separate profile/port and
recheck ownership before testing there. Its existing service was not changed.
