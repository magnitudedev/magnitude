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

## Windows removal recovery and command lifetimes, 2026-09-23

Parent: `16f27d52`. Fixed uninstall after an update retained a mapped previous payload. After the
installed uninstaller is matched against its executing self-copy, removal retires the previous tree
against its inventory before changing current payload or registration. Unknown or still-mapped files
stop removal and preserve the current installation. Empty remnants use the same bounded retirement
rule as interrupted-install recovery. Extraction scratch is cleared only after previous retirement.

The foreground launcher now classifies `serve` separately from finite commands. Serving retains the
atomic kill-on-close job and one startup continuation; finite commands forward ordinary process
lifetime and cannot request continuation. This preserves an independently launched desktop when
its CLI caller exits. The serving bootstrap must verify native containment when it is integrated;
release packaging and stable-launcher maintenance remain pending.

Executed on Windows ARM64 with x64 runtime/launcher and x86 installer helper, ordinary user:

- MSVC `/W4 /WX` builds passed. Native job and launcher tests passed, including an independent process
  surviving its finite CLI caller and preservation of that caller's nonzero exit status. This is a
  process-lifetime surrogate, not the required real Electron `app open` regression.
- The real production NSIS fixture passed A→B→C continuation. Before each continuation, an unknown
  previous file and the mapped old runtime separately made uninstall fail while preserving current
  version, PATH and the installed removal record. After continuation, uninstall removed both current
  and retained payloads. Fresh reinstall and another uninstall then passed.
- Native interrupted replacement passed after old-directory movement, before registration commit,
  and after commit. Each scenario also repeated recovery successfully.
- Installer-rendering suite passed all 29 tests on pinned Bun 1.4.2. No TS production APIs changed.

The uninstall fixture runs a byte-identical self-copy directly with NSIS's explicit installation
argument, retaining that process's actual result rather than the asynchronous bootstrap's result.
Each copy has a unique name so immediate retries do not overwrite an image still being released.
An initial ad-hoc native-test build omitted its current-user manifest and triggered elevation;
that attempt was cancelled without approval, and the test was rebuilt with the normal asInvoker
manifest. No acceptance result relies on an elevated test run.

Artifacts are in `removal-acceptance-2/` beneath the previously recorded isolated Windows directory.
The actual installed app and uninstall registration are absent after acceptance. The subsequent
native recovery tests may leave an empty private extraction container, which is expected scratch.
No fixture processes remain. The former retained-payload evidence was removed through a successful
installer recovery and exact uninstaller operation, not an unverified product cleanup path.

Linux now has a separate source checkout of the prior checkpoint under
`/home/trg.guest/magnitude-headless.uAol17lC/checkout`; dependency setup is underway independently of
the running installed baseline. Next implementation work can extract shared bootstrap while remaining
platform update admission and launcher packaging gates continue to be exercised.

### Phase 3 work in progress — shared bootstrap and output

Extracted application profile, matched resources/service command and supervised startup composition
from desktop main into daemon-management. Desktop retains its existing ownership, UI and lifecycle.
Added explicit diagnostic-only versus foreground child output on both native spawners. Collection
retains the last 16 KiB; terminal forwarding allows one bounded outstanding write and cannot hold
shutdown open. Added installed payload canonicalization and a real filesystem symlink-chain test.
These changes are not yet a completed phase checkpoint.

Evidence on Mac with pinned Bun 1.4.2:

- The focused bootstrap/output/Unix-child/Windows-composition/supervisor/port suite passed 26 tests;
  the subsequent symlink-resolution addition passed all six bootstrap tests.
- A real owned child and its worker retired after a deliberately failing foreground stderr sink.
  Blocked writes, late asynchronous terminal errors, synchronous write errors and bounded diagnostics
  passed. Windows composition initially rejected the new extra command field; explicit native command
  construction corrected that and all three composition tests passed.
- Daemon-management and desktop targeted typechecks passed after output integration. The later
  canonicalization addition still needs its final typecheck.
- The extraction's built Electron app ran against isolated profile
  `/tmp/magnitude-headless-bootstrap.9kK09M`, port 11163. Real GUI Status showed Ready; source CLI
  service status and models status succeeded without serve. CLI app open succeeded and the UI
  remained Ready. A close-button action was performed, but accessibility immediately showed a
  window again, so this run does not independently establish a hidden-window interval.
- Explicit CLI service stop completed; the retained Electron execution session exited 0 and no
  processes with that profile/port remained. The personal installed application was not replaced.

Remaining Phase 3 gates include native Linux/Windows execution of the extraction, stronger window
close/reopen observation, final builds/typechecks, and resolving the existing full-suite shell-probe
failure. The Windows launcher still needs release packaging; cross-platform update admission and
packaged end-to-end tests remain open. No foreground serve command is implemented yet.

Follow-up verification: daemon-management typecheck also passed after canonicalization. Synced the
current extraction into the separate Ubuntu ARM64 checkout, rebuilt its native addon, and passed
all 27 focused tests there, including actual Unix child/worker retirement and the failing-terminal
case. This is native Linux process coverage, not packaged Linux desktop acceptance.

### Shared bootstrap checkpoint verification

Corrected a test-runtime attribution error: adding the pinned Bun directory to PATH did not replace
`bunx`, because that directory originally contained only `bun`. The resolved `bunx` was a symlink to
the user's Bun 1.3.14. Earlier Mac test claims of Bun 1.4.2 based only on that PATH override were
incorrect. Their observed results stand, but those runtime labels are superseded by this verification.
Windows and Linux commands that explicitly invoked the pinned `bun x --bun` were unaffected.

Created a local bunx symlink beside the isolated pinned runtime and verified it reports 1.4.2.
The previously failing shell-probe suite passes all seven tests on that runtime; temporary
instrumentation was removed and no shell-probe implementation or test was changed. Full reruns:

- Mac desktop: 256 passed, six platform/integration skips; all 32 executed files passed.
- Mac daemon-management: 263 passed, seven platform skips; all 37 files passed.
- Desktop targeted typecheck and production bundle/native build passed after the final changes.
- Windows focused suite: 21 passed, one Unix symlink test skipped. Native compiled ACN was then
  exercised by both Node 24.21.0 and Bun 1.4.2 owners using the changed Windows spawner; both received
  final startup-failure health, acknowledged it, observed exit and retired the native job.

This is a tested implementation checkpoint for shared bootstrap extraction, not a claim that the
full Phase 3 visual/package matrix has passed. Actual model interaction, settings persistence/login
registration in a disposable desktop profile, stronger close/reopen observation, and packaged
three-platform acceptance remain explicit gates alongside subsequent ownership integration.

### Phase 4 work in progress — owner contract and cooperative arbitration

The SDK snapshot now carries Desktop-with-tray or Headless ownership, and the intent schema includes
Yield. Existing desktop producers, renderer/CLI readers, control fixtures and loading fixture were
migrated together, without accepting the previous snapshot shape. Explicit app Open over a Headless
snapshot launches Desktop and waits for Desktop observation rather than treating Headless as a window.

Acquisition now accepts an explicit Desktop or Headless request. Headless contention fails before
contacting the incumbent. Desktop observes the current owner, forwards to Desktop or requests Yield
from Headless, then retries native acquisition under one 60-second bound. Cold/closing missing
endpoints permit retry; access, malformed-message and other control failures remain errors. No path
unlinks the lock, signals an incumbent process or treats Yield acknowledgement as transferred ownership.

Mac pinned-runtime evidence: all 35 focused client/control/owner tests passed, including real native
lock exclusion and Unix IPC handoff, Desktop forwarding, both snapshot schema forms, rejection of the
old shape, and Windows transport simulation of reply-before-dispatch for Yield. An additional assertion
proves the newly acquired lock remains held after the contender fiber returns. Daemon-management,
CLI and desktop targeted typechecks passed. This remains uncommitted Phase 4 work: the serving runtime,
platform installation admission, signal lifecycle and actual native server takeover are not yet added.

### Phase 4 work in progress — first foreground serving execution

Added lazy public serve registration and its privileged runtime. The runtime composes shared profile
and resource selection, signal observation, native ownership, installation exclusion, service
supervision and owner control without Electron. It reports Headless snapshots, rejects login/update
requests without launching a desktop, acknowledges Yield before stopping, and propagates terminal
service/cleanup failure. Production Windows admission validates native parent job containment.
Startup update reconciliation is still a later phase; the interim update refusal is explicit.

Linux gained a separately tagged scoped native shared installation capability, opening the fixed
root-owned read-only lock with close-on-exec. On the Ubuntu VM, gracefully stopped the previously
running installed 0.1.5 desktop via its CLI. The VM installation is now stopped. Rebuilt the modified
addon and ran the native lease fixture with both Node and pinned Bun: shared admission excluded an
independent exclusive flock; an exec'd child inherited no installation descriptor; forged/cross-kind
release was rejected; repeated correct release was safe; exclusive flock succeeded after release.
This is native admission evidence, not installed serve or installer-race acceptance.

Mac source execution used isolated `/tmp/mag-serve-phase4`, port 11164 and the pinned Bun path.
`magnitude serve` reached Ready without Electron. Source CLI models status read that running service;
a second serve failed with exit 1 and left the first serving. Ctrl+C requested administrative ACN
shutdown and the foreground execution exited 0; no profile/port-matching processes remained in the
subsequent process listing. CLI and daemon-management targeted typechecks passed. Help printed the
new command without importing its runtime.

Still open before the Phase 4 checkpoint: initial port preflight before supervision, earlier-signal
admission tests, actual desktop takeover with the rebuilt schema, Windows foreground native execution,
installed/display-free Linux serve, failure and contention race coverage, full process-identity
retirement evidence and packaged acceptance. The native Linux installation fixture currently requires
an installed lock and no active owner; it is not part of the portable unit suite.

### Phase 4 follow-up — actual Mac handoff and display-free Ubuntu serving

Rebuilt the desktop with the new owner schema. Started isolated Mac serve on port 11164, then ran
source CLI app open against that same profile. The headless execution exited 0 after administrative
shutdown. Its recorded owner 44966, ACN 44967 and inference 44973 were absent afterward. The desktop
reported Ready/Registered on the same endpoint; computer-use navigation and screenshot verified the
actual Status screen. Explicit CLI stop then shut down that isolated desktop. This proves a real
handoff, though continuous race-timeline instrumentation and packaged takeover remain open.

Separated port preflight from its spawner wrapper. Foreground bootstrap now retires previous installs
and checks the port before supervision, while Desktop retains supervised admission failures; every
child attempt still checks the port. A real occupied-port source serve exited 1 with the expected
message in 0.177 seconds. The 12 focused bootstrap/port/ownership tests passed. Signal observation
now starts before headless admission and a pending stop can prevent service construction after
platform checks.

Synced current source to the separate Ubuntu checkout and ran serve with DISPLAY and WAYLAND_DISPLAY
removed, isolated profile `/tmp/mag-serve-phase4`, port 11164, and the installed 0.1.5 engine manifest
as an explicit development override. It reached Ready and source CLI models status succeeded.
Recorded Linux process start identities for owner 31577, ACN 31592, inference 31612 and four planning
workers. Sent SIGTERM followed immediately by SIGINT to the verified owner. Every recorded identity
retired within the bounded observation and the retained SSH command exited 0. No GUI owner was
started. This is source-runtime native coverage, not packaged installed Linux acceptance.

### Phase 4 follow-up — compiled native Windows serving

Built the current CLI/service using the production build functions and rebuilt the native Windows
launcher. The disposable VM now has a matched serving fixture at the normal Local AppData
Programs/Magnitude/resources path: compiled magnitude.exe, magnitude-service.exe and desktop-host.node.
The native launcher remains outside that payload in serve-acceptance. This directory is a serving
fixture, not a complete installed desktop or a package acceptance receipt; no desktop installer or
registration was produced in this step.

Normal-user native launcher execution with isolated profile serve-acceptance/profile and port 11164
reached Ready using the existing 0.1.5 engine manifest as an explicit test override. Compiled CLI
models status succeeded, a second contained serve exited 1 without replacing the owner, and compiled
CLI service stop succeeded. Recorded the launcher/CLI/ACN/inference/worker tree before shutdown;
all 15 recorded processes disappeared. The original launcher test session then completed with exit 0.
The VM control tool had retained its command session until the long-running child stopped even
though the PowerShell script had already exited; no duplicate server was started to recover it.

Added lazy-runtime coverage for serve and subprocess coverage for serve help and rejected port,
data-dir and host flags, proving those paths leave the chosen profile absent. All 21 entrypoint/lazy
boundary tests passed on the verified pinned Mac runtime. A separate actual Mac serve with a missing
engine completed bounded restart attempts and exited 1 after 17.27 seconds with its final failure.

Remaining Phase 4 gates still include compiled-owner crash/signal acceptance, continuous contention
and takeover race evidence, installed Linux admission/race refusal, complete installed Windows
desktop takeover, and packaged three-platform regression. The new Windows serving fixture is stopped;
its payload and logs remain for further acceptance work.

### Foreground ownership implementation checkpoint verification

Forced the actual native Windows launcher to exit after the compiled serving payload reached Ready.
All nine recorded launcher/CLI/service/inference/worker identities retired; comparison included native
creation dates. The fixture also ensured its exact launcher was terminated on any test failure.

On Mac, eight simultaneous source serve subprocesses shared a fresh isolated profile and port 11167.
Exactly one remained serving and the seven contenders exited 1. Forced that verified owner to exit;
its recorded seven-process tree disappeared, including inference planning workers. This exercises
actual concurrent startup and parent-loss cleanup. It does not substitute for continuous OS event
coverage of multiple simultaneous desktop takeovers.

Added native lock tests for 32 concurrent acquisition attempts (one admission, 31 refusals) and a
virtual-clock deadline test retaining an unresponsive incumbent's native lock after the contender
fails. Full pinned-Bun Mac suites passed: daemon-management 271 (seven platform skips), CLI 77,
desktop 256 (six platform/integration skips). Existing desktop shell-probe tests remain green.

This is a substantial working-code checkpoint for foreground serving and cooperative takeover.
The plan remains intentionally open for packaging/launcher integration, native Windows console
cancellation with the real server, installed three-platform takeover/race acceptance, Linux busy
installer/marker refusal, actual desktop model/settings/login regressions, and the final full matrix.
Continue CLI cutover and shared-update implementation while retaining those explicit acceptance gates.

Final targeted daemon-management, CLI and desktop typechecks passed. Corrected the new virtual-clock
test's separate Effect layer provisions to one combined provision; all six native owner tests passed
again. No broad regression failures remain in this checkpoint's executed unit suites.
