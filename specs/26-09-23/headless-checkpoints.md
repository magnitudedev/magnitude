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
| 0: baseline and harness | Baselines recorded; final regression gates remain | Native/source baselines and live desktop checks recorded below |
| 1: Windows preparation repair | Implemented; final hosted acceptance pending | `f9b148d7`, `fdcd30a2`; native staging, cache repair and retained-update removal exercised |
| 2: native continuation/admission proof | In progress | `f3762f0b`, `16f27d52`; repeated Windows replacement passed; macOS mechanical probe passed, production integration open |
| 3: shared owner extraction | Implemented; packaged regression carried forward | `18345064` |
| 4: serve and takeover | Implemented; full packaged race acceptance carried forward | `cadd4bdd`; real foreground serving on three platforms and live Mac takeover |
| 5: CLI cutover | Implemented; final packaged regression carried forward | `41ec9f6b`; 97 CLI tests passed on each platform, live Mac desktop-owned CLI exercised |
| 6: shared updater/macOS transaction | In progress | `c2906a6c`; shared preparation extracted, native macOS transaction still open |
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


### Phase 5 work in progress: connect-only clients and passive status

Removed the SDK CLI starter and its command-specific errors. Pi model commands now explicitly use
connect-only SDK admission. Ordinary CLI service-backed operations observe the application owner
before connection and report the agreed no-service message on confirmed absence. Other control
failures remain errors. Added 14 real subprocess cases for hardware/catalog/model/connection
operations: each exits 1 with that message and leaves a fresh absent profile untouched.

Replaced the public service namespace with passive status, retaining the hidden native runtime
probe. Status returns success for absence, displays Desktop/Headless ownership, and omits desktop-only
tray/login fields for Headless or absence. Existing unavailable active-model presentation is retained.
The absent-status subprocess fixture initially exceeded the macOS control socket path limit; the
command correctly rejected that invalid path. Shortening the fixture prefix fixed its test.

Executed on Mac with pinned Bun 1.4.2: SDK 40 tests, Pi 101 tests, CLI 97 tests all passed;
SDK/Pi/CLI targeted typechecks exited 0 (existing Effect language-service advisory messages remain).
Pi build exited 0. No remaining SDK/Pi/CLI references to the deleted starter APIs were found.

This is not yet a checkpoint: migrate native desktop acceptance scripts and workflow commands,
remaining public documentation, and the development Pi launcher before committing Phase 5.
Broaden passive status coverage to live owners and complete the phase's existing-desktop regression.


Phase 5 continuation: migrated public command documentation and the Windows native CLI smoke
workflow to status/app open. Linux installed lifecycle now exercises explicit Desktop launch,
passive status, owner Quit, and separate login disable; Windows hosted-update acceptance uses the
actual tray Quit action before reopening. Both modified JavaScript fixtures passed Node syntax
checks; these updated installed fixtures still require native execution against new packaged builds.

The Pi development launcher now explicitly ensures its development Desktop before acquiring the
connect-only client. Its six script tests passed with scripts/vitest.config.ts and the targeted
scripts/tsconfig.dev-pi.json check exited 0. The initial root test invocation selected no tests;
it was corrected to use the scripts configuration, not counted as validation.

Live Mac source regression used isolated /tmp/mag-serve-phase4 and port 11164. Foreground serve
reached Ready; status reported Headless with no tray/login fields and models status succeeded.
Explicit app open took over and the retained foreground session exited 0. Status then reported
Desktop, Ready and Registered tray; hardware and models status succeeded against that Desktop.
Computer use visibly exercised Discover and Settings, selected Light then Dark and restored System.
Development login/updates correctly remained disabled; no installed login acceptance is claimed.
The real app Quit shortcut exited the isolated Desktop. Subsequent status reported Stopped/None,
and models status returned the no-service message without relaunching it. Actual model inference,
packaged login controls, and cross-platform acceptance remain open gates.


### Phase 5 connect-only implementation checkpoint verification

Ubuntu aarch64 and Windows x64 Bun 1.4.2 CLI suites both passed all 97 tests. Native Windows testing
found the pre-existing subprocess helper used URL.pathname, producing an invalid /C:/ path. Replaced
it with fileURLToPath and tightened rejected-syntax assertions so entrypoint resolution failures
cannot pass as command rejection. Mac's corrected 32 subprocess cases passed again. The Windows
suite passed after correcting the expected Commander excess-argument diagnostic for removed commands.

Built the Windows CLI with the production build function and matched native addon. Compiled status
returned Stopped and models status returned the exact no-service error; both left a new absent profile
untouched while LOCALAPPDATA intentionally named an invalid location. Initial temporary harness
attempts hit PowerShell 5 quoting/stderr handling and a missing native-library environment input;
corrected harness invocation and final smoke exited 0. No production workaround was introduced.

Mac full Desktop regression: 256 tests passed, six platform/integration tests skipped. Client-common
connection/lifecycle/presentation: ten tests passed. SDK protocol tests retain exact mismatch rejection
and connect-only admission. Final CLI targeted typecheck exited 0. Earlier no-tests-selected desktop
invocation is not counted; the full Desktop suite above is the executed check.

This checkpoint closes the implemented CLI cutover, docs/workflow migration and source/compiled CLI
checks. Updated Linux installed-lifecycle and Windows hosted-update fixtures remain pending execution
against the final matched packages; they are not certified by syntax checks or source tests. Full
packaged regression, real model serving and login acceptance remain tracked by the later gates.

Final targeted client-common and Desktop typechecks also exited 0.


### Phase 6 work in progress: shared preparation

Moved the application update engine, schedule, installation identity, hosted transfer source and
Linux metadata reader from Desktop into daemon-management's application-update export. Their
existing tests moved with them. The preparation engine consumes the SDK state contract directly,
without depending on client-common. Hosted transfers capture injected filesystem/path services;
Desktop supplies NodeContext and the moved tests run with BunContext. No Electron dependency or
implicit application launch is present in the shared preparation modules.

The existing state machine and durable store remain authoritative: one admitted transfer, scoped
cancellation cleanup, preserved prepared bytes, failed-attempt visibility and separate scheduling.
Updated the governing update design and applicability. Native platform installation adapters still
live in Desktop pending the installation-intent and native transaction work.

Pinned Bun Mac validation: shared updater 17 tests passed; Desktop preparation/platform/preference
15 tests passed. Daemon-management and Desktop targeted typechecks exited 0 after wiring the host
filesystem/path capabilities at composition. Custom macOS replacement, installer outcomes, headless
startup application, finite maintenance and full platform acceptance remain unfinished.

Desktop production-mode local build, including native adapter and renderer, exited 0 after extraction.


Shared installation extraction now includes the prepared-attempt barrier and Linux/Windows adapters.
Installation intent is a Schema separating authorization from Desktop visibility or Caller continuation.
Windows/Linux helper relaunch paths return without spawning an application for Caller. This is an
explicit continuation boundary, not completed headless update integration: foreground lifetime,
completion observation, new-version execution and cancellation still need their Phase 7 wiring.
The existing macOS adapter explicitly refuses non-Desktop intent pending its native replacement.

Mac targeted validation: 30 shared update tests passed (four Windows-only skips), 17 Linux/Windows
handoff tests passed, and six remaining Desktop macOS/preference tests passed. Windows VM ran the
13 engine and 12 prepared-installation tests successfully, then all four native private-file staging
cases passed after correcting moved fixture imports and the native addon path. Those fixtures use
real ACL handling but do not claim hosted download or publisher-signature end-to-end acceptance.
Daemon-management, Desktop and CLI targeted typechecks exited 0. Desktop build exited 0 with shared
adapters and continuation intent. No running service now performs automatic installation; that
integration remains deliberately unfinished until native transactions and foreground continuation
are validated.

Full daemon-management regression passed: 303 tests, 11 platform/integration skips. This is a
shared-update extraction checkpoint within Phase 6; native macOS and full installation gates remain open.

### Phase 6 work in progress: macOS transaction mechanics

At parent `c2906a6c`, added an isolated native macOS fixture and a dedicated native-workflow job.
The fixture compiled locally with warnings treated as errors and a macOS 13.0 deployment floor,
then exited 0. It copies its own executable into temporary directories; it does not modify an
installed application or authorize production installation.

Observed checks:
- Process loss at four journal/exchange boundaries preserves distinguishable old/new directory
  identities. Repeated reconciliation does not reverse an exchange.
- Missing, symlinked, substituted and ambiguous directory identities refuse reconciliation.
- Execution through the replaced path retains the same PID, Unicode arguments, environment,
  working directory and open standard descriptors.
- The deliberately inherited installation lock excludes an independently opened contender until
  release; a separate close-on-exec descriptor is not inherited.

The CI job is configured but has not been executed remotely. Local output is reproducible by the
compile/run commands in that job. These are mechanical checks, not signed application acceptance:
archive containment, publisher validation, native production capability adoption, cross-user
admission, cancellation, sync failures and power-loss durability remain open. The current local
keychain reports no valid code-signing identities, so production signed-bundle validation requires
the protected signing path; no trust fallback was added. The prototype is included in the native
bundle-verification checkpoint below.

### Phase 6 checkpoint: native macOS bundle verification

Parent source `c2906a6c`. Added a Security.framework verifier with an asynchronous Node-API boundary
and an Effect service. Production service construction requires the compiled publisher; no runtime
environment or update-response field chooses trust. Verification checks the signed bundle identity,
sealed version, application package type, Mach-O executable, requested architecture, nested code,
resources and all architecture slices. The caller must retain exclusive staging ownership throughout
verification and publication. This check is not yet wired into a production installation transaction.

The universal-binary fixture exposed a native API distinction: creating a code object with an explicit
architecture restricted verification even with the all-architectures flag. The verifier now validates
an unqualified code object and separately requires the intended architecture. A universal fixture with
one correct slice and one incorrectly identified slice now fails; both correct slices pass.

Validation on the local Mac with pinned Bun 1.4.2:
- Native build passed with warnings treated as errors and macOS 13.0 deployment target.
- Twelve focused tests passed, including sealed framework version symlinks, damaged nested code,
  changed resources, absent signatures, incorrect identity/version/architecture, embedded NUL,
  universal slices, and refusal to construct production trust without a compiled publisher.
- Full daemon-management regression: 315 tests passed, 11 platform/integration skips.
- Daemon-management, CLI and Desktop targeted typechecks exited 0.
- Mechanical exchange/recovery/foreground probe passed again; workflow YAML parsed successfully.
- Node independently verified the actual installed signed 0.1.5 bundle using its observed publisher
  requirement, then rejected a deliberately incorrect publisher. This was read-only; no installation
  or running application was changed. It verifies native signed-input behavior, not release provenance
  or an installed update transaction.

The native workflow now builds and runs bundle-verification fixtures as well as the mechanical probe;
remote execution remains pending. Production signed replacement, notarization/Gatekeeper behavior,
private extraction, exclusion, recovery and continuation integration remain open. Desktop compilation
must provide publisher identity when the new service is integrated; the existing CLI release compiler
already supplies that build constant. No existing desktop update backend was switched in this checkpoint.

### Phase 6 work in progress: private macOS archive extraction

At parent `7a5b9869`, added a standalone extraction helper using the operating system archive engine.
Public API headers are pinned with source, checksums and retained license notices. It extracts only
into an empty current-user private directory and confines paths to one application root. Entries and
expanded output are bounded; unsafe paths, parent-relative/absolute link targets, duplicate entries,
privilege modes, writes through links, dangling links and cycles fail. Normal framework version links,
executable modes and macOS metadata survive. The system ZIP reader may interpret unsupported special
mode attributes as ordinary files; the helper never creates device nodes, FIFOs or other special files.

Local evidence:
- Warnings-as-errors build and Clang static analysis passed with macOS 13 deployment target.
- Fifteen native extraction fixtures passed, then passed again with address and undefined-behavior
  sanitizers. Cases include corrupted content checksums, truncated ZIPs, path/type conflicts, duplicate
  writes, private/empty-directory admission and a real `ditto` extended-attribute round-trip.
- Read the installed signed 0.1.5 application, archived it with the production ZIP flags, and extracted
  into `/tmp/magnitude-signed-extraction.qvvQKB/stage`. Recursive content comparison, strict nested
  signature verification and the new native publisher/version verifier all passed on the copy.
  The installed application was not modified or launched.
- Added the extraction build/fixtures to the macOS native workflow; remote execution remains pending.

This helper is not yet assembled into releases or invoked by startup. The transaction must authenticate
and retain the archive, retain installation exclusion, own staging cleanup and durability, verify the
extracted bundle, and authorize replacement. Cancellation/parent-loss containment and inherited native
capability integration remain open. Release assembly must include the vendored header license notices
when adding the helper. The deployment flag is not a substitute for actual macOS 13 execution.

### Phase 6 checkpoint: staging and durable filesystem primitives

Parent source `7a5b9869`. Added native retained directory capabilities, an Effect filesystem service,
bounded private record reads, durable atomic record publication and identity-checked bundle exchange.
These synchronous bounded operations are for the finite installer process, not the desktop event loop.
Capabilities revalidate parent identity and permissions; private directories additionally reject
extended ACL grants. Record replacement refuses symlinks, hard links and unsafe existing objects.
Record publication syncs contents before rename and the parent after rename, with full filesystem
flushes. Exchange verifies both expected identities, uses descriptor-relative atomic exchange, checks
the resulting identities and syncs both parents. Callers must reconcile an exchange error because it
may follow a successful namespace mutation. No exchange retries are hidden inside the primitive.

Native build and Clang static analysis passed. Nine actual native filesystem tests cover record
replacement, identity-preserving exchange, stale replay refusal, parent substitution, missing versus
unsafe entries, extended ACL grants, changed permissions, hard links, size limits and capability
tagging/release. Initial execution exposed the no-ACL `ENOENT` result from the native ACL API; that
case is accepted, while actual grants and other observation errors fail. Full daemon-management
regression passed: 324 tests, 11 platform/integration skips. Targeted package typecheck exited 0.
The extraction fixtures and sanitizer results from the preceding entry are included in this checkpoint.
The macOS workflow runs both native filesystem and signature tests; its YAML parsed locally.

These primitives do not yet implement the schema-validated transaction journal, recovery state
machine, rollback decisions, installation-wide exclusion, staged-tree durability or startup execution.
Production transaction fault injection and power-loss acceptance remain open; the earlier mechanical
probe does not prove those future integrations. No automatic replacement has been enabled.

### Phase 6 checkpoint: interrupted exchange recovery

Parent source `145dcb7d`. Added the schema-validated, identity-bound transaction journal and recovery
state machine. Exchange intent can become committed, abandoned, or restoration intent; restoration
intent can become restored. Terminal states cannot authorize another forward exchange. Recovery
verifies the installed bundle before declaring it usable, verifies the displaced old bundle before
rollback, and never rolls back a committed replacement. Unknown identities, malformed records and
failed reconciliation return a repair-required error. Native directory capabilities now expose their
canonical path and explicit parent synchronization for post-crash namespace durability.

Evidence on the local Mac:
- Forty-five recovery tests passed, covering all twenty journal-state/observed-layout combinations,
  replay, missing/unknown identities, parent/name binding, invalid UTF-8 and malformed journals,
  invalid replacement and invalid rollback bundle, and completion sync/record failures.
- Five fixture subprocesses were actually terminated with SIGKILL after intent publication, forward
  exchange, commit publication, restoration-intent publication and restoration exchange. The native
  filesystem operations and durable record writes were real; parent recovery then ran twice and
  preserved the expected version without toggling the exchange.
- Recovery fixtures inject bundle-verification decisions to isolate state transitions. Actual
  signature verification has separate native evidence above; these tests do not claim an integrated
  signed updater transaction. Injected operation failures before/after mutation likewise do not
  substitute for filesystem or VM power-loss testing.
- Full daemon-management regression: 369 passed, 11 platform/integration skips. Targeted package
  typecheck, native build, Clang static analysis and workflow YAML validation passed.

Recovery still requires caller-held installation exclusion. Initial preparation, staged-tree sync,
archive authentication/capability transfer, transaction cleanup, fresh-install publication, foreground
execution and startup integration remain unfinished. This checkpoint does not enable automatic
replacement or authorize service startup through an uncertain transaction.

### Phase 6 working results: transaction preparation and installation admission

Uncommitted work above `a27c35d9`; retained for the integrated installer checkpoint, following the
requested larger checkpoint cadence. The execution-plan checkpoint guidance now reflects that cadence.

The initial exchange transaction verifies both versions, synchronizes the staged tree, durably
publishes intent, exchanges once and reconciles actual identities. Cancellation before publication
leaves no intent; after publication it waits for reconciliation. Native staged-tree synchronization
is bounded, does not traverse symlinks, and refuses special files and hard-linked files. A signed
fixture integration test now exercises the real native verifier, tree sync, exchange and recovery
together using an explicitly injected ad-hoc test requirement. Production trust remains unchanged.

Added installation-wide shared/exclusive native admission with a scoped Effect capability. The
stable lock is adjacent to the bundle, owned by its owner and readable across users, and remains
outside bundle exchange. File creation is exclusive and permitted only to the bundle owner. Existing
lock files are never repaired, replaced or unlinked. Validation rejects unsafe modes, extended ACLs,
links, nonempty files, substituted parent/lock identities and forged native capabilities. Descriptors
are close-on-exec. New admission is not yet wired to owner startup or replacement; installations owned
by another user require the installer to provision the lock. Prior-version owner exclusion remains a
separate migration gate, as does actual cross-user acceptance.

Local Mac evidence (Bun 1.4.2, arm64):
- Recovery/initial transaction tests: 54 passed, including interruption and failures on both sides of
  exchange. Native filesystem tests: 11 passed. Actual signature tests: 13 passed.
- Installation admission: seven tests passed, including independent-process contention and SIGKILL
  release without lock replacement, shared-reader/exclusive-writer exclusion, bundle replacement,
  unsafe files, extended ACLs and retained capability validation.
- Full daemon-management suite: 388 passed, 11 platform/integration skips, exit 0. Log:
  `/tmp/magnitude-headless-transfer/phase6-mac-admission-suite.log`.
- Targeted daemon-management typecheck and native build exited 0. Clang static analysis passed for
  changed transaction filesystem and admission sources. `git diff --check` passed.

These results do not prove a packaged startup update. Admission migration, retained archive transfer,
extractor containment, cleanup, foreground continuation, startup composition, signed release assembly,
actual cross-user/system-manager and abrupt VM shutdown acceptance remain unfinished. No automatic
replacement has been enabled, and no checkpoint commit was made for these intermediate results.

### Phase 6 working results: authenticated extraction composition

Still uncommitted above `a27c35d9`. The native build now produces the extraction helper alongside the
existing command-lifetime helper. A shared Effect staging service authenticates the release for the
exact Mac ZIP target, invokes extraction through the native command guard, and revalidates staging
before returning. The extractor hashes its retained no-follow archive descriptor before and after
parsing, checks the authenticated byte count, and rejects write access for other users, hard links
and extended ACLs. Private staging now rejects extended ACLs at the extraction boundary too.

The existing native signed-fixture transaction now starts with a publisher-signed ZIP release and
runs guarded extraction, native bundle validation, staged-tree synchronization, exchange and recovery.
An invalid release signature and modified archive bytes both fail before populating staging. This
is an ad-hoc bundle/test-publisher integration fixture, not production signing or packaged startup.
Native build, 13 bundle/integration tests, seven admission tests, targeted package typecheck and
extractor Clang static analysis passed. Eighteen Python extraction cases passed normally and with
address/undefined-behavior sanitizers. `git diff --check` passed. The full-suite result in the preceding
entry predates these extraction changes; only affected tests were rerun here.

Migration investigation rejected using an empty whole-machine process-search result as admission
proof: the implementation can omit failed observations. Direct executable-path observation detected
an unrelated live process with an unresolved executable on this Mac. Consequently, whole-machine
path scanning also cannot provide a practical migration guarantee without a separate identity model.
The exploratory observation code was removed; no process was terminated. The shared kernel lease
remains, while prior-version migration policy and its acceptance remain open. Automatic replacement
is still disabled. Integration must not silently equate an unobservable process to an absent owner.

### Phase 6 working results: terminal cleanup and repeated signed transactions

Uncommitted work remains above `a27c35d9`. Added descriptor-relative displaced-tree removal and
exact-content journal removal. Cleanup first requires a terminal transaction and revalidates the
installed bundle through recovery. It removes only the expected displaced identity within private
staging, never traverses symlinks, and keeps the journal throughout partial deletion. The exact
terminal record is removed and durably synchronized last. An unsuccessful deletion is a distinct
cleanup failure, not a claim that the installed bundle needs rollback. Empty-journal retries sync
the staging directory. Old terminal receipts cannot remain eligible for recovery after a later
transaction changes the installed identity.

The actual signed fixture now performs two successive publisher-authenticated ZIP extractions,
native signature checks, exchanges and cleanup, verifying versions 0.1.5 → 0.1.6 → 0.1.7. Both
transaction directories are empty afterward. This remains fixture-bundle acceptance, not a packaged
Desktop/CLI/ACN update or production publisher acceptance.

Added terminal-state cleanup/replay, refusal of nonterminal and unjournaled deletion, installed-bundle
validation before retirement, partial-deletion failure/retry, outside symlink/hard-link preservation
and exact receipt matching tests. Three additional fixture subprocesses actually die by SIGKILL
after partial cleanup, displaced-tree removal and receipt removal. Recovery observes the expected
version and cleanup completes without another exchange. These are process-loss tests, not VM
power-loss evidence.

Native build, targeted daemon-management typecheck and Clang filesystem static analysis passed.
Full daemon-management suite passed: 401 tests, 11 platform/integration skips, exit 0; log
`/tmp/magnitude-headless-transfer/phase6-mac-cleanup-suite.log`. The first full-suite invocation used
the workspace directory accidentally; it was interrupted (exit 130), preserved separately as
`phase6-mac-cleanup-wrong-scope-interrupted.log`, and is not a claimed pass. The corrected package run
is the result above. `git diff --check` passed. No checkpoint commit was made; startup/continuation,
transaction discovery, migration exclusion and packaged acceptance remain unfinished.

### Phase 7 working results: finite update preparation and maintenance admission

Uncommitted alongside the Phase 6 integration work. Added shared finite preparation and persisted
observation, independently of the long-lived owner's timer. Passive observation reads only prepared
state and preferences. Check never auto-downloads, even when the saved preference is enabled.
Download performs one check and waits through durable preparation plus transfer-scope retirement;
it refuses to claim Ready if staging did not publish the exact unattempted release. Existing
prepared/failed installers are not silently replaced or retried. Discard waits for store cleanup.
These operations do not install an application or change preferences.

A scoped maintenance entry acquires the same native application lock without a control listener,
service or takeover, then rechecks the per-user installer lease before any operation. Contention
fails immediately. The normal ownership directory setup is shared with application admission.
The public CLI has not yet been routed to this entry; update-source/configuration composition,
owner routing, finite installation completion and headless scheduling remain to integrate.

Nine finite-preparation tests and eight native owner-arbitration tests passed (17 total). They cover
passive reads, check without auto-download, publication/cleanup ordering, cancellation during staging,
missing publication, retained installation failures, failed discard, owner/maintenance contention,
installer exclusion and ownership release. Targeted daemon-management typecheck and diff whitespace
checks passed. Mac native evidence does not establish Windows/Linux maintenance acceptance. No commit
was made; the next checkpoint remains substantial integrated behavior with its full validation.

### Phase 7 working results: installed preparation and CLI routing

Uncommitted. Desktop build configuration now has a shared strict decoder and a packaged
`update-configuration.json` resource for installed CLI preparation. Status and discard do not acquire
a network source or request identity. The CLI passively observes the owner, routes once to a present
owner, and uses scoped maintenance when absent. An owner mutation's lost reply is never retried via
maintenance. The native host update method no longer ensures or launches Desktop. Ready output
identifies whether the user must stop a live foreground server first.

Targeted validation: 19 shared tests passed (configuration 8, installed preparation 2, finite
preparation 9), plus 8 CLI routing/output tests. Installed preparation uses real temporary resources
and missing profiles with write capabilities that fail if called; both Mac and Windows target
composition remains observational and isolated production source acquisition refuses before writes.
These are Mac-hosted tests, not Windows packaged acceptance. Targeted daemon-management and CLI
TypeScript checks passed. An initial Windows fixture used an unsupported arm64 release target and
was corrected to the shipped x64 target. Lazy-import mocking did not intercept maintenance under the
Bun test runtime; the absence routing test now exercises the actual development-build refusal.

The command matrix is not complete: finite install still refuses without an owner, the headless
owner's update endpoint remains to integrate, and new resource assembly needs a Desktop build and
packaged verification. Startup installation, scheduling, continuation and platform acceptance remain
open. No checkpoint commit was made.

### Phase 7 working results: live headless preparation controls

Uncommitted. Headless bootstrap now initializes installed preparation after ownership/installation
admission and before service startup. It composes the shared update engine and persisted preferences,
reconciles retained records, and retains one scoped schedule. Control accepts status, check, download
and discard; install returns stop-first guidance with no shutdown or installer capability. Ready
state changes print restart guidance in the foreground terminal. Development builds remain explicitly
unavailable for updates. Initialization failures leave serving available with an unavailable update
state. This connects preparation only; startup installation remains unfinished and the reported
restart path still needs the platform continuation work before feature acceptance.

Twelve targeted tests passed: live-headless control 2, schedule 2, native ownership 8. They exercise
read-only status, preparation requests, install refusal without close, timer finalization, and native
owner exclusion. CLI and daemon-management typechecks passed with existing informational Effect
language-service diagnostics. These results do not replace live installed-server or packaged
cross-platform acceptance. No commit was made.

Full CLI suite subsequently passed: 10 files, 102 tests, exit 0, including passive entrypoints,
connect-only ordinary commands, status, update routing and presentation. Log:
`/tmp/magnitude-headless-transfer/phase7-cli-suite.log`. Whitespace validation passed.

### Phase 6/7 working results: packaged preparation resources

Node-driven Desktop build passed and emitted the shared update configuration. Desktop and release
package typechecks passed. Assembly now includes the macOS extraction executable; signing assigns
it the native-helper profile without JIT entitlements. An isolated application was assembled under
`/tmp/magnitude-headless-transfer/phase7-assembly`, using the existing installed 0.1.5 CLI and service
as read-only assembly inputs. It was not launched and is not a matched-current-runtime acceptance
fixture. The personal installation was not modified.

The assembled configuration equals Desktop's generated configuration and its publisher fields match
packaged trust. Ad-hoc signing through the release signing function and strict deep verification
passed. The helper's signed entitlements are empty; loader inspection lists only system libarchive
and libSystem. All 18 extraction fixtures passed against the executable inside this signed bundle.
An attempted post-signing byte comparison with the unsigned build input failed because signing
changes executable bytes; it was not counted as validation. Signing and native behavior were then
verified directly. Production Developer ID/notarization and matched-version application update
acceptance remain open. Logs: `phase7-desktop-build.log`, `phase7-desktop-types.log`,
`phase7-release-types.log`, `phase7-assembly.log`, `phase7-sign.log`, and
`phase7-packaged-extraction.log` under `/tmp/magnitude-headless-transfer`. No commit was made.

### Phase 6/7 working results: startup integration review

The existing Linux Desktop handoff explicitly waits for the parent's lifetime pipe to close before
starting package installation. A foreground caller cannot wait synchronously on that same handoff;
its installation/completion path must be separate, retain invoking-user identity and authorize the
narrow installed package operation before continuing. No foreground completion claim was added.

Desktop now substitutes its Apple publisher Team ID at build time under the same Developer ID mode
used by CLI compilation. Missing/empty and malformed Team IDs were both exercised through actual
Electron build configuration loading and failed with the intended error. The normal ad-hoc Desktop
build passed. Runtime environment cannot supply this compiled value; the custom verifier still
refuses an absent production identity. The verifier is not yet connected to Desktop replacement.
Also added a stop-request recheck after headless update initialization so cancellation during that
initialization cannot proceed into service startup. No commit was made.

### Checkpoint: shared preparation and macOS transaction integration foundation

Checkpoint includes the substantial uncommitted preparation/control, native extraction/admission,
transaction cleanup, packaged resources and associated design work described above. It does not mark
Phase 6 or 7 complete. The unfinished Linux foreground installer is excluded and retained separately
for continuation. Remaining startup/continuation and packaged acceptance gates remain authoritative.

Final checkpoint validation on macOS arm64 with Bun 1.4.2:
- Shared runtime: 424 passed, 11 platform/integration skips, exit 0.
- CLI: 102 passed, exit 0.
- Desktop: 227 passed, 2 packaged/integration skips, exit 0 on full rerun. The initial concurrent
  run returned empty shell output in one unchanged harness-quoting test; its isolated 11-test file
  and the full rerun passed without changes. The intermittent failure remains recorded, not diagnosed.
- Targeted shared-runtime and CLI typechecks passed; Desktop/release typechecks and Node Desktop
  builds passed earlier in this checkpoint. Native and packaged extraction evidence is recorded above.
- Diff whitespace check passed. Added text was checked for prohibited sensitive references.

Final logs under `/tmp/magnitude-headless-transfer`: `phase7-checkpoint-daemon-tests.log`,
`phase7-checkpoint-cli-tests.log`, `phase7-checkpoint-desktop-tests.log` (initial failure),
`phase7-checkpoint-desktop-retest.log`, `phase7-checkpoint-daemon-types.log`, and
`phase7-checkpoint-cli-types.log`. This checkpoint is not cross-platform production acceptance.

### Phase 7 working results after c5fb3742: Linux foreground completion

Uncommitted. Explicit no-owner Linux install now retains maintenance plus per-user installation
admission, verifies the retained archive, durably records the attempt, and waits for sudo to run the
installed privileged package entry. Interactive stdin/stderr permit terminal authorization; otherwise
sudo uses noninteractive mode. Completion requires a successful helper exit and the installed CLI
reporting the prepared release before discard. Installer/version failures retain failed state;
interruption retains the attempted record. No Desktop is launched. The existing root-only installed
entry now accepts sudo's invoking-user identity as well as its existing authorization identity,
rejecting absent, invalid, root and conflicting identities before touching the request.

Nineteen targeted tests passed on Mac (8 completion/order/failure/cancellation, 11 authorization
identity). On the Ubuntu arm64 VM the same tests plus 7 privileged package-verification tests passed
(26 total). These mock package execution; actual sudo/package replacement is still an open gate.
VM was reachable, noninteractive sudo returned uid 0, installed CLI reported 0.1.5 and no service
process was observed. Shared-runtime and CLI typechecks passed after resolving the expanded Effect
requirements in CLI composition. Log: `/tmp/magnitude-headless-transfer/phase7-linux-native-tests.log`.
Startup exec/continuation, cancellation during an actual package transaction and authorization-denied
VM acceptance remain unfinished. No additional commit was made.

### Phase 7 working results: real Linux finite package transaction

Ubuntu arm64 disposable VM now exercised the compiled CLI's actual no-owner `update install` path
through sudo, the root-only installed helper, publisher/hash/package identity verification and apt.
Compiled A reported 0.1.5. The signed prepared package contained compiled B reporting 0.1.6; the
finite command exited 0 and printed installation completion only after B version verification.
The prepared record was retired, the package-manager transaction marker was absent, and no Desktop
or service process was observed. The original VM application payload is retained under
`/home/trg.guest/magnitude-headless.uAol17lC/phase7-package/original`; the Mac installation and Sparky
were untouched. The installed VM is now a disposable 0.1.6 package fixture, not a production release.

Package: `magnitude-desktop_0.1.6-1_arm64.deb`, 193336508 bytes,
SHA-256 `e16f4ec88af1fe7ec485a729eac3562d73c762a566a70f987835140f49ce1b05`.
The fixture reused the installed graphical/service payload with the newly compiled CLI to isolate
this transaction. It does not prove matched-release Desktop or inference acceptance. Local ephemeral
publisher trust was installed only in this test VM. Initial packaging failed because an installed
payload omits the packager's expected LICENSE input; the fixture used the installed copyright file
and the corrected packaging run passed the normal package identity/permission validation.

Logs under `/tmp/magnitude-headless-transfer`: `phase7-linux-cli-build.log`,
`phase7-linux-package-build.log` (initial fixture failure), `phase7-linux-package-rebuild.log`,
`phase7-linux-real-install.log`. Startup exec, actual transaction interruption, denied authorization
and repeated A→B→C remain open. No checkpoint commit was made.

### Phase 7 working results: Unix foreground replacement

Uncommitted native continuation performs exec with explicit bounded arguments/environment, rejects
embedded NULs and invalid environment entries, and leaves the caller alive if exec fails. It never
uses a shell for dispatch. The Effect adapter exposes failure without introducing a waiting parent
or detached replacement. Native ownership descriptors remain close-on-exec; replacement reacquires
normal admission before starting service work.

Native builds and two process-level tests passed on Mac arm64 and Ubuntu arm64. The fixture executes
the real Bun runtime again and verifies equal PID, exact literal argument, cwd, explicit environment,
stdin and stderr preservation, and successful acquisition of the old process's ownership lock after
exec. Malformed input and nonexistent executable tests return errors without terminating the caller.
Linux evidence: `/tmp/magnitude-headless-transfer/phase7-linux-continuation.log`; Mac build:
`phase7-unix-continuation-build.log`. This primitive is not yet wired to startup installation;
package-manager cancellation and same-invocation A→B→C serving remain open. No commit was made.

### Phase 7 working results: real Linux startup installation and continuation

Uncommitted startup composition now runs after application ownership but before the shared Linux
installation lease and service creation. It considers only unattempted retained releases, checks
noninteractive sudo authorization for the exact installed helper, and defers without changing the
record if unavailable. Authorized startup retains the per-user installation lease, waits for verified
completion, then executes the replacement CLI with the original invocation/environment. The native
continuation adapter is loaded before replacement. Stop signals race startup work; a completed stop
cannot proceed to service admission. Failed installation/exec propagates rather than starting an
uncertain old runtime. Both affected package typechecks passed.

Executed on Ubuntu arm64 using a real generated deb, current compiled CLI A 0.1.5 and matched CLI/
service B 0.1.6. A single `magnitude serve` invocation (PID 36493) installed B, retired the prepared
record and exec'd the installed replacement without changing PID. Health then reported service
0.1.6, revision 46, RPC 2, Ready (service PID 36784); ordinary CLI status reported Headless Ready
0.1.6. The first service startup log followed apt completion. SIGTERM returned exit 0; subsequent
process observation found no Desktop/service and the package transaction marker was absent.
Installed package is now fixture 0.1.6-2. The graphical payload was retained from the original VM
installation and was not launched; this proves headless update continuation, not graphical release
acceptance or inference generation.

Package `magnitude-desktop_0.1.6-2_arm64.deb`: 193340462 bytes,
SHA-256 `fc7ffb03497ab628886aa9cc28a35b295a672ac16660342421dd4e86ab479c4f`.
VM fixture `/home/trg.guest/magnitude-headless.uAol17lC/phase7-startup` retains build inputs and log.
Host logs: `/tmp/magnitude-headless-transfer/phase7-linux-startup-build.log` and
`phase7-linux-startup-acceptance.log`; type logs `phase7-startup-cli-types.log` and
`phase7-startup-native-types.log`. Repeated B→C, denied authorization, cancellation during package
replacement and system-manager lifetime tests remain open. No commit was made.

### Phase 7 working results: denied Linux authorization and repeated startup update

The Ubuntu fixture now passed the second real startup replacement, B 0.1.6→C 0.1.7, following the
previous A→B run. A temporary validated sudoers rule denied only the installed update-helper command.
With C prepared, startup retained the Unattempted record and served B (foreground PID 39854,
service PID 39866, service identity ik7pg9vhiman). A live `magnitude update install` returned the
stop-first failure; subsequent health preserved both service PID and identity. No update ran while
that server was alive. The denied server stopped cleanly.

After the temporary rule was removed, the next invocation (PID 39938) installed C and exec'd the
replacement under that same PID. Service Ready reported version 0.1.7, revision 46, RPC 2, PID 40230;
the prepared record was retired. Shutdown returned 0. Independent follow-up confirmed the test rule
was absent, the package transaction marker was absent, installed CLI reported 0.1.7 and no Desktop/
service process remained. Native replacement identity was checked against `/proc/<foreground>/exe`.

Package `magnitude-desktop_0.1.7-1_arm64.deb`: 193340452 bytes,
SHA-256 `48c47d248e2a270a3a0d738b837943da894ee639b537c686f6bcad4a26d1c9ca`.
Logs: `/tmp/magnitude-headless-transfer/phase7-linux-c-build.log` and
`phase7-linux-denied-and-c.log`. These use matched compiled CLI/service fixtures and the retained
original graphical payload; no graphical or model-generation claim. Actual mid-install cancellation,
system-manager cleanup and full release acceptance remain open. No commit was made.

### Phase 7 working results: Linux service-manager lifetime

Real Ubuntu user-systemd acceptance passed with KillMode=control-group, Restart=no and a bounded
stop timeout. A hardened unit with NoNewPrivileges=yes could not authorize installation, retained
the Unattempted 0.1.8 preparation and reached Ready on 0.1.7 (MainPID 40425, service 40439). A normal
unit then installed 0.1.8 and continued with unchanged MainPID 40509; service 40817 reported Ready
0.1.8. Both foreground and service processes were verified in their respective unit cgroups.
Stopping each unit returned Result=success and ExecMainStatus=0; captured PIDs disappeared. Follow-up
showed both units inactive, no package transaction marker, and installed CLI version 0.1.8.

Package `magnitude-desktop_0.1.8-1_arm64.deb`: 193340460 bytes,
SHA-256 `5823163973aa7d0a7c06e586edfcc558347051f72c0f513c076f1b1bc32431ce`.
Logs: `/tmp/magnitude-headless-transfer/phase7-linux-d-build.log` and `phase7-linux-systemd.log`.
This covers normal completion and managed shutdown after readiness, plus hardened-unit authorization
deferral. It does not cover interruption during package mutation, which remains open. No commit.

### Phase 7 checkpoint: Linux startup installation and bounded cancellation

Implemented finite Linux installation and startup-only installation before service admission. The
replacement preserves the foreground PID and invocation. Startup defers without prompting when
system authorization is unavailable; live servers reject installation. Explicit installation verifies
both the signed package and the installed CLI version before retiring preparation.

A paused real deb pre-install script exposed two cancellation defects: privileged descendants outlived
the foreground process, and a never-ending command-input stream retained the foreground runtime.
The privileged installer now retains a caller lifetime pipe; package commands run under the native
command supervisor. Linux supervision adopts and reaps descendants even when package tools create
new sessions. Scoped input completion releases the command-input pump. Earlier failed fixture runs
were repaired through dpkg before further acceptance; they are not counted as passing tests.

Final interruption acceptance used systemd MainPID 42853 and captured eleven cgroup processes during
actual package mutation. Stop returned Result=success and ExecMainStatus=0. All captured processes
retired within 33 ms after stop returned. Preparation remained Attempted, the installation marker
remained present, and the prior CLI remained 0.1.8. This deliberately requires package-manager repair;
it cannot be silently retried as an unattempted update. The test allows up to three seconds for native
asynchronous descendant retirement and checks actual process state, rather than relying on unit state.

After repairing the disposable VM, the final code installed fixture 0.1.9 from 0.1.8 in one startup.
Foreground PID 43010 continued into the installed replacement; service PID 43316 reached Ready with
version 0.1.9, revision 46, RPC 2. The prepared record was removed only after successful verification.
Ordinary CLI status reported Headless Ready 0.1.9. SIGTERM returned 0. Package control version is
0.1.9-6; fixture filename remains magnitude-desktop_0.1.9-1_arm64.deb, 169554508 bytes,
SHA-256 c47f4c5e4a244c9765c6d0aff70bafba8cc077db5b11dc8952627038ecd21799.
The graphical payload is retained fixture content; this does not prove graphical release acceptance.

Validation: macOS daemon-management 446 passed / 12 platform skips; CLI 102 passed; Desktop 227 passed /
2 skips. Linux targeted suite 30 passed, including real native same-PID exec and escaped-session child
retirement. Both affected package typechecks exited 0 with existing advisory diagnostics. macOS and
Linux native builds passed. Logs are under /tmp/magnitude-headless-transfer: phase7-daemon-regression.log,
phase7-cli-regression.log, phase7-desktop-regression.log, phase7-linux-final-tests.log,
phase7-linux-interruption-bounded.log and phase7-linux-final-startup.log.

This is a substantial Linux checkpoint, not completion of all update work. Interactive authorization
with terminal policies, complete RPM acceptance, macOS installer integration, Windows launcher
packaging/integration, full release acceptance, live GUI regressions and remote inference remain open.

### Phase 6 integration in progress: macOS owner installation admission

Installed Desktop and Headless entry points now acquire the tested shared bundle installation lease
before service initialization and retain it in their owner scope. Development remains outside this
installed-bundle admission. Exclusive admission fails owner startup before update initialization;
normal cancellation releases the shared lease. The fixed adjacent lock permits different profiles
to participate in the same installation exclusion. Inaccessible or unsafe admission fails closed;
installation provisioning and migration remain required before full release acceptance.

Native headless integration plus existing native admission tests passed: 8 tests. The integration test
runs actual headless owner admission against a disposable bundle and native addon, proves that an
exclusive installer prevents initialization, proves the admitted owner excludes installation, then
interrupts the owner and proves release. Desktop targeted typecheck exited 0. The first daemon-management
typecheck identified the test fixture missing its SQLite layer; the fixture now provides the ordinary
Bun SQLite layer, and its typecheck was rerun.
Logs: /tmp/magnitude-headless-transfer/mac-owner-admission-tests.log,
mac-owner-admission-desktop-types.log, mac-owner-admission-native-types.log. These changes remain
uncommitted pending the larger macOS integration checkpoint; custom installer handoff, transaction
discovery, prior-owner migration and packaged/GUI acceptance are still open.

### Phase 6 integration in progress: discoverable macOS installation workspace

Added one stable private transaction workspace adjacent to the installed bundle. Opening it retains
exclusive installation admission, binds the staging entry to its native directory identity, and
synchronizes its discovery entry before staging or journal publication. Observation does not create
staging. Exchange rechecks lease identity at the filesystem capability boundary. Unpublished partial
extraction can be cleared for an explicit retry; a published receipt must go through reconciliation.
Native scope cleanup removes only the exact empty private workspace and syncs the parent. Nonempty,
unsafe, unrelated or substituted entries are preserved.

Prepared-install orchestration now composes saved archive verification, durable Attempted state,
archive staging, verified exchange, recovery and terminal cleanup. It refuses an active prior native
installation. Preparation retirement precedes removal of the committed transaction receipt: a failed
preparation discard can be reconciled without another extraction or exchange. Cleanup failure after
commit remains a cleanup warning rather than authorizing rollback. Actual helper/startup wiring and
prior-version exclusion are still open; this does not enable automatic macOS replacement yet.

Native regression run: 89 tests passed across transaction filesystem, recovery, workspace, prepared
installation and headless admission. Follow-up expanded workspace/prepared-install tests: 12 passed,
including substituted-directory protection and partial-extraction failure followed by explicit retry.
The native addon rebuild passed. Targeted daemon-management typecheck passed after mapping lease
validation failures into the filesystem capability's declared error channel. Logs are under
/tmp/magnitude-headless-transfer: mac-workspace-native-build.log, mac-workspace-regressions.log,
mac-workspace-completion-tests.log, mac-workspace-final-types.log. These tests use real filesystem
exchanges and kernel leases with an injected bundle verifier/stager; production publisher, extraction
and packaged acceptance remain separate gates. No additional checkpoint commit was made.

### Phase 6 integration in progress: foreground installer admission transfer

Added explicit macOS exclusive-lease transfer across exec. Preparing continuation duplicates only the
exclusive installation descriptor for inheritance; ordinary descriptors remain close-on-exec. A fresh
installer validates the inherited descriptor against the named lock and parent, consumes it, and
retains a close-on-exec capability. Failed exec closes the temporary descriptor while retaining the
original lease. Shared leases cannot prepare this transfer. Workspace composition accepts a retained
exclusive capability only for its bound installation, so helper adoption need not reacquire its own
already-held lock.

A native process fixture executed original → installer → replacement with one PID, proved shared
admission remained excluded inside the installer, and proved the replacement could acquire ordinary
shared admission after the installer descriptor closed on exec. Repeated failed exec and invalid
adoption inputs were also tested. Native rebuild and targeted daemon-management typecheck passed;
22 lease/workspace/prepared-install tests passed. Logs: /tmp/magnitude-headless-transfer/
mac-lease-handoff-build.log, mac-handoff-workspace-tests.log, mac-handoff-workspace-types.log.

These remain integration primitives. Creating/authenticating the external helper, invoking the real
CLI hidden entry point, automatic startup recovery, prior-owner migration and packaged publisher/GUI
acceptance are not yet complete. No additional checkpoint commit was made.

### Phase 6 integration in progress: private helper and hidden installer command

Added scoped preparation of the external macOS installer runtime. It verifies the installed bundle,
copies the CLI/native addon/command supervisor/extractor into private storage, checks every copied
code file against the compiled publisher policy, and writes the decoded installed update trust
configuration. Unsafe existing helper storage, linked runtime files, bundle-verification failure and
copied-code verification failure refuse preparation; incomplete files are cleaned up. No unsigned
publisher fallback was added.

Added a bounded schema-checked hidden CLI installer entry point. Its request must match the executing
private helper path and contain a valid inherited descriptor; continuation is finite completion or
foreground serve arguments, never an arbitrary executable. Composition adopts the installation
lease, acquires application maintenance, runs the prepared transaction using copied resources,
retires the exact helper directory by native identity, then returns or execs the installed CLI.
The foreground starter composes helper preparation and lease-preserving exec. It remains unwired to
public startup/maintenance until prior-version exclusion and packaged acceptance are established.

Validation: 27 helper-preparation/invocation tests passed; ordinary CLI regression suite 102 passed;
daemon-management and CLI targeted typechecks exited 0. Logs under /tmp/magnitude-headless-transfer:
mac-helper-command-tests.log, mac-helper-command-final-types.log, mac-installer-cli-types.log,
mac-installer-cli-regression.log. Helper preparation tests inject signature verifiers; they establish
copy/permission/cleanup orchestration, not production publisher acceptance. The hidden command has
not yet passed a signed end-to-end packaged installation. No checkpoint commit was made.

### macOS helper integration checkpoint and signed CI preparation

The integrated chunk now contains shared owner admission, discoverable transaction workspace,
prepared installation, exclusive lease transfer across exec, private helper preparation and the hidden
CLI installer. Public automatic startup and finite maintenance are deliberately not connected until
migration policy and signed acceptance are established. This checkpoint does not mark Phase 6 complete.

Expanded native CI now runs the macOS helper/admission/continuation suites and the Linux foreground
update/descendant-retirement suites. Added an explicitly dispatched signed macOS installer job in the
existing Apple signing environment. Its new harness builds two complete signed/notarized applications
with the standard bundle identity, a temporary publisher key and a loopback-only update origin; it
publishes no release or hosted offer. It invokes the copied helper and real hidden CLI entry point,
then checks replacement version, preparation/helper/transaction retirement, native signatures,
notarization ticket and Gatekeeper. The producer accepts a supplied fixture configuration without
changing existing hosted acceptance defaults. This signed job has not yet run successfully.

Also corrected native updater observation for a user with no GUI launchd domain. On this Mac a
lookup of a nonexistent GUI domain returned code 112 with its exact missing-domain diagnostic;
that specific response now means absence. Unrelated diagnostics, another UID and other failures
remain errors. Nine observation/headless admission tests passed.

Local checkpoint validation: daemon-management 490 passed / 12 skips; expanded macOS CI selection
155 passed / one Linux-only skip; CLI 102 passed in the preceding integration run; targeted
daemon-management, CLI and release typechecks exited 0. Workflow YAML parsing and diff whitespace
checks passed. Logs: /tmp/magnitude-headless-transfer/mac-helper-checkpoint-regression.log,
mac-helper-checkpoint-types.log, mac-helper-checkpoint-release-types.log, mac-expanded-ci-tests.log,
mac-no-gui-observation-tests.log. Local keychain still has zero valid signing identities. The remote
Apple signing environment has no branch restriction or additional approval rule, so branch CI is the
available production-signature path. A one-time reboot migration restriction has been raised as a
product preference; no migration bypass or unproven process-absence assumption was enabled.

A substantial checkpoint is being committed to let signed CI test this exact source tree. Signed
end-to-end results, public startup/recovery wiring, migration, remaining platforms and full GUI/remote
inference acceptance remain open.

Checkpoint: cf9bf329, Connect macOS installer admission and private helper execution. The initial
publication f21cdf57 failed workflow parsing before jobs ran because runner.temp was placed in a
job-level environment field. That configuration correction was folded into the same checkpoint;
actionlint 1.7.12 then passed. The branch update used an exact expected-SHA lease.
Signed/native validation was successfully dispatched as GitHub Actions run 35947492258 against
cf9bf32992c6ffc4189f4fd81ca02ff24cb720e4. Both Linux jobs passed; macOS failed because
the fresh checkout lacked generated protocol identity, and Windows failed in the cache ACL fixture.
The signed installer job was skipped because its native prerequisite failed.
Run URL: https://github.com/magnitudedev/magnitude/actions/runs/35947492258.


### Native CI fixture corrections within the macOS integration checkpoint

Added version generation to both macOS CI setup paths. Repeated the expanded native selection in a
fresh detached checkout using the corrected setup: 156 passed, one Linux-only skip. Actionlint and
whitespace validation pass.

The Windows cache fixture previously inherited arbitrary temporary-directory permissions. It now
creates a parent with the recognized user/SYSTEM/Administrators inheritance and explicitly assigns
the current-user owner to the child, independent of elevated-token defaults. The child ACL itself is
still inherited. Recovery failure diagnostics now print the returned native status. Production ACL
validation and recovery remain unchanged. On the Windows VM, the revised test compiled with /W4 /WX
and passed. In a temporary directory with an additional Everyone grant, the old test reproduced the
CI failure (exit 1 at cache retirement) while the corrected test passed (exit 0). This also preserves
unknown-content refusal, old-byte retirement and repeatability assertions.

These CI setup/fixture corrections are folded into the existing integration checkpoint to avoid a
separate small checkpoint. Signed acceptance still requires a successful rerun.

### Startup recovery separation and second native CI pass

Checkpoint 85e893ab is running as Actions run 35948213286. macOS native acceptance and both Linux
jobs passed. Windows passed native security/cache recovery, jobs, launcher lifetime and pipe tests,
then exposed the same ambient temporary-ACL assumption in the Node/Bun junction fixture. That fixture
now uses a native-created private parent and explicitly sets child ownership while preserving actual
ACL inheritance. Both Node 24.21.0 and the pinned Bun runtime passed on the Windows VM. Production
permission/recovery logic is unchanged. This correction remains uncommitted pending the next checkpoint.
The signed macOS job is confirmed running; no signed installation success is claimed yet.

Added a separate macOS prepared-transaction recovery operation. It observes existing staging without
creating a workspace and reconciles published transactions without admitting another attempt or
requiring the archive extractor. Partial extraction with no receipt remains for explicit retry;
failed preparation stays failed. Tests now retain preparation state across operations and prove
committed receipt reconciliation, no replay after failed extraction, and no installation merely from
unattempted preparation. Combined recovery/workspace/installer tests: 102 passed. Targeted
package typecheck exited 0. Logs: /tmp/magnitude-headless-transfer/mac-prepared-recovery-tests.log
and mac-recovery-types.log. These changes remain uncommitted; public startup wiring is still open.

Signed macOS acceptance in run 35948213286 completed successfully against 85e893ab: the real copied
helper/hidden CLI installed the 0.0.502 app, retired preparation/helper/transaction storage, and passed
codesign, stapler and Gatekeeper (Notarized Developer ID). Log retained locally as
/tmp/magnitude-headless-transfer/mac-signed-installer-ci.log. Artifact 10786909601 contains the two
signed ZIPs and result receipt; retrieval is in progress. This proves finite A-to-B installation,
not public startup, desktop continuation or migration. The local acceptance harness now additionally
requires A-to-B-to-C and matching CLI/service versions after each replacement; targeted release
checking passes, but the expanded signed scenario has not run yet. Node and pinned Bun junction
fixtures also passed under SYSTEM, in addition to the normal-user runs. No new checkpoint was made.

The private installer request now requires an explicit Install or Recover operation. Recover uses
transaction reconciliation without constructing an archive stager and permits continuation after a
verified preservation result; Install still fails if the requested replacement did not commit.
Missing/unknown operations are rejected before admission. The signed harness now includes a finite
recovery invocation after each of its two replacements. Native transaction/workspace and invocation
tests passed (105); daemon-management and release targeted typechecks passed; ordinary CLI regression
passed. Logs: mac-recovery-command-tests.log, mac-recovery-command-types.log,
mac-recovery-release-types.log and mac-recovery-cli-regression.log under the local transfer directory.
Public startup/migration is not enabled by this private-command change. No additional commit yet.

### macOS foreground startup wiring and signed local regression

Connected installed macOS serve startup and finite no-owner installation to the private custom
installer. Published receipts select recovery first; otherwise only unattempted preparation selects
installation. Recovery receipt retirement is required before exec continuation to prevent an exec
loop on cleanup failure. The first upgraded application arrives through the existing updater's
quit/relaunch path; no reboot or extra migration ceremony is required. Startup-selection tests passed
(6); macOS application-update tests passed (37); ordinary CLI tests passed (102); targeted CLI and
daemon-management checking passed after completing the test-store implementation. Full signed
acceptance of this new public wiring is still pending; these edits remain uncommitted.

Downloaded artifact 10786909601 and inspected its successful finite-install receipt. Extracted the
signed 0.0.502 app to /tmp/magnitude-headless-transfer/signed-mac-local, leaving the personal
installation untouched. CLI and service version commands both returned 0.0.502. Using isolated
profile /tmp/magnitude-headless-transfer/signed-mac-profile and port 11237, with the local development
ICN installation override, serve reached Ready. Launching the same signed desktop cooperatively
retired Headless (exit 0), and CLI status then reported Ready/Desktop. Computer use opened the real
Status page and visually verified Ready at the isolated address. The bundled ordinary models status
command succeeded against the desktop-owned service. Quit was driven through the real desktop UI;
this run proves packaged ownership/CLI/GUI regression, not model inference or the new uncommitted
startup-update wiring.

### macOS foreground installation checkpoint

The next signed harness exercises the public finite update-install command for A-to-B and public
serve startup for B-to-C, waits for a Ready Headless service while retaining the original command
process, then requires graceful zero-exit shutdown. Each replacement verifies matched CLI/service
versions, private storage retirement and publisher/Gatekeeper acceptance. This expanded signed run
is pending; the prior A-to-B private-helper acceptance remains the completed signing evidence.

The full local daemon-management run initially had one intermittent lease assertion failure amid
499 passes. That suite passed in isolation (9 tests), and a complete repeat passed 500 tests with
12 platform skips. Logs: mac-startup-checkpoint-tests.log and mac-startup-checkpoint-repeat.log.
The new startup path also defers unattempted installation when the application parent is not writable,
retaining the update and printing permission guidance. Recovery cannot take that deferral path.
This substantial checkpoint connects macOS foreground installation/recovery and finite maintenance;
Desktop updater replacement, full signed public-command acceptance and remaining platform work are
still open. It does not declare the whole phase complete.

### Desktop custom installer integration checkpoint

Signed run 35949905805 completed public finite A-to-B at b127a73a, but the later artifact-log audit
showed serve-startup B-to-C failed before Ready because the fixture version had no published ICN.
The pipeline masked that failure; its green step was not valid startup acceptance. Log: mac-signed-public-installer-ci.log in
/tmp/magnitude-headless-transfer. Native Mac and both Linux jobs passed. The Windows job failed
because its fixture inherited a PowerShell module path incompatible with the child shell. The
fixture now uses the .NET ACL API directly; Node and pinned Bun passed in the disposable Windows
VM with an intentionally unavailable module path. Production directory validation is unchanged.

Desktop now prepares the same archive as the CLI and closes its owned resources before entering the
shared installer. Startup handles pending installation/recovery before starting a service; explicit
installation preserves window visibility through the private helper. Removed the replaced staging
and handoff implementations. The signed harness adds a third replacement through Desktop startup;
that new signed continuation scenario remains pending at this checkpoint.

Desktop typechecking and production bundle assembly passed. Desktop tests passed 222 with two
platform skips; daemon-management passed 498 with 12 platform skips. Targeted release checking
passed. Logs: desktop-custom-update-tests.log, mac-desktop-shared-installer-tests.log and
 desktop-update-release-types.log under the transfer directory. An initial shell test failure under
local Bun 1.3.14 was reproduced and diagnosed; it passes with the project-pinned Bun 1.4.2, including
the full Desktop suite. Temporary instrumentation was removed without changing shell behavior.
This checkpoint completes the Desktop wiring, not final signed Desktop acceptance or the remaining
Windows/package-script/remote inference gates.

### Remote inference acceptance on Sparky

At 2336c140, copied the branch into /home/tom/magnitude-headless.KY79vO5a and built native ownership
there. Source CLI serve reached Ready on isolated port 11237 and profile using the existing installed
ICN runtime; the stale development ICN declaration was first rejected for a missing binary. Copied
the existing model cache into the isolated profile, restarted the test owner, and loaded
qwen3.6-35b-a3b:gguf:q4 through the ordinary CLI. A real OpenAI-compatible chat request produced a
response (109 generated tokens, approximately 63 tokens/second). Local receipt:
/tmp/magnitude-headless-transfer/sparky-headless-inference.json. Models stop unloaded it; SIGTERM to
the verified test owner exited zero, CLI status reported Stopped/None and port 11237 was closed.
The original user magnitude.service remained active throughout. This proves source headless ownership
and inference on the DGX, not a full script-installed package or startup update on that machine.

### Windows release integration in progress

Run 35950727523 passed the repaired Node/Bun private-cache fixture and Rust child retirement. Its
next failure was the NSIS handoff fixture still sending showWindow instead of the required Desktop
continuation. Updated that request and ran the full native/NSIS installer test on the Windows VM:
configuration preservation, interrupted recovery, fresh install, unknown-file refusal, owner-exit
handoff, upgrade, startup preservation and uninstall passed. A prior unregistered test resources
folder was preserved under the test root before running the fixture. The native launcher compiled
successfully with the actual Windows toolchain (153600 bytes).

Uncommitted release wiring now builds and bundles the launcher alongside the matched CLI and includes
it in signing and installer payload validation. Windows installer contract tests passed (29), and
targeted release checking passed. External launcher installation, PATH maintenance and foreground
startup continuation remain unfinished; these packaging edits are not a phase checkpoint.

Signed macOS run 35950727523 produced four signed fixture applications and completed finite CLI
A-to-B. The downloaded log contradicts its green step: B-to-C exited before Ready after trying to
acquire ICN for an unpublished fixture version. Desktop C-to-D was therefore never exercised, and
result.json was absent. The earlier success report based on the job step was incorrect. Both signed
startup and Desktop continuation gates remain open. The harness now resolves an explicit real ICN
installation; its pipeline propagates failure and requires the final receipt. Artifact ZIPs were
retrieved to /tmp/magnitude-headless-transfer/signed-mac-35950727523 for local GUI regression.


A disposable Windows probe tested FileRenameInfoEx with replace/POSIX flags against its own mapped
executable. The first replacement returned ERROR_ACCESS_DENIED (5); it made no product changes.
Do not rely on atomic replacement of an executing launcher. Launcher maintenance must explicitly
account for a mapped image. Probe source is /tmp/magnitude-headless-transfer/windows-launcher-replace-probe.c.


### Phase 7 Windows foreground update implementation and acceptance

The working tree now installs a private native command outside the replaceable application,
registers its PATH entry, and retains the original foreground launcher across one verified replacement.
Finite installation releases application ownership while retaining installation admission; startup
only attempts unattempted preparations and never replaces a live server. Missing update configuration
defers automatic installation without preventing serving; explicit installation still reports it.
The six Windows orchestration tests passed on the VM. CLI regression tests passed (102), and targeted
CLI/release type checks passed. Full native/NSIS fixtures passed before the added interruption case.
Repeated native testing exposed a loader-startup race in the mapped-image fixture; it now waits for
an explicit child readiness event. The expanded fixture also checks repair between launcher renames.
The expanded native DLL suite passed, including interrupted launcher publication repair and all three
application replacement recovery states. The signed packaged finite/startup update harness is still running.

The packaged Windows harness uses a separate native DLL copy, since loading the build output before
rebuilding it pins that DLL on Windows. Its real ICN fixture resolves published 0.1.5. The VM output
uses a short directory to avoid the test root exceeding Windows staging path limits. The temporary
acceptance publisher is scoped to this disposable VM and must be removed after signed testing.

The signed 0.0.504 macOS fixture was launched from a disposable /tmp application and isolated profile.
Computer use verified Discover, Ready Status and Settings; ordinary CLI status, models status and
hardware queries worked against that Desktop owner. The app quit with exit 0. The personal installed
application was not replaced. This is desktop regression evidence, not signed update continuation.


Packaged Windows acceptance now has an opt-in native CI entry with temporary test-publisher cleanup
and a required completion receipt. Workflow lint and the PowerShell parser passed. Its maintained
harness also opens Desktop after the update sequence, checks Ready plus ordinary model/hardware
commands after the finite launcher exits, and quits the isolated owner. This added regression is
not yet runtime evidence; the currently running VM harness predates those final checks.

The first complete Windows cohort built and signed all three packages and installed 0.0.501. It then
failed in fixture preparation because the isolated profile parent had not been created. Native
private-directory creation is intentionally nonrecursive. The harness now creates its private profile
before staging; direct native creation of profile and updates passed. No application update was
proved by that run. A fresh cohort is necessary because its in-memory publisher key was lost; future
runs now retain public signed release records before installing anything, without retaining a private
signing key. The rerun writes a complete VM log rather than relying on truncated terminal output.

Before removing 0.0.501, its external native command successfully opened Desktop and returned while
Desktop remained alive. Status reported Ready, Desktop, 0.0.501, and the isolated endpoint; models
status and hardware both succeeded. Computer vision observed Discover render detected CPU and memory.
Navigation through Parallels did not change views, so this is not proof of interactive Windows UI
operation. Native Quit completed and a separate CLI status confirmed Stopped/None; uninstall then
passed. A harness wait was corrected to observe Stopped instead of expecting a retained endpoint
record to disappear. The fresh signed sequence, including maintained desktop CLI checks, is running.

### Phase 7 checkpoint: Windows packaged continuation and native RPM recovery

Windows signed packaged acceptance completed with exit 0. The receipt at
`/tmp/magnitude-headless-transfer/windows-headless-result.json` records installed 0.0.503,
finiteInstall, foregroundContinuation, gracefulExit and desktopCliRegression. The complete log is
`/tmp/magnitude-headless-transfer/windows-headless-acceptance.log`; VM artifacts remain at
`C:\Users\trg\hu-f77c57f4`. The sequence installed 0.0.501, used the public command for finite
0.0.502 installation, applied 0.0.503 before foreground serving, observed Ready while the original
launcher remained alive, and stopped cleanly. It then opened Desktop with the public command,
observed Desktop Ready, ran models status/hardware, quit, and observed Stopped. Both temporary test
publisher trust and its signing key were removed after verification. This is disposable test-publisher
acceptance, not production Windows publisher certification. Interactive Windows navigation remains
unproven through the current computer-use adapter; macOS live UI operation has been exercised.

A Fedora 44 arm64 VM (`magnitude-rpm`, Lima VZ, two CPUs, 4 GiB) now provides native RPM coverage.
Its test uses the existing 0.1.9 compiled Linux fixture and the current package scripts, not a newly
built final release cohort. Revisions 7 and 8 installed/upgraded and served with published CPU ICN
0.1.5. An upgrade during serving was refused while the old server stayed Ready. A refused removal
exposed DNF removing automatic dependencies despite the RPM pre-uninstall refusal. RPM pre-uninstall
now leaves the existing repair gate on refusal; it does not interrupt the live owner. Subsequent
startup reports package-manager repair until reinstallation restores dependencies and clears the gate.

The maintained `test-linux-rpm.sh` passed on Fedora with revisions 7 and 9: fresh installation,
upgrade refusal with live model queries, clean SIGTERM, stopped upgrade, restart, removal refusal
with continued live queries, stopped startup refusal with repair guidance, dependency-restoring
reinstallation, Ready plus hardware, clean stop, successful removal, and preserved profile sentinel.
The exit-0 receipt and logs are under `/tmp/magnitude-headless-transfer/rpm-evidence`; VM evidence is
`/tmp/magnitude-rpm-acceptance.yAnfJaB2`. No test owner remains. RPM interrupted-transaction fault
injection and final-cohort desktop/inference coverage are still required beyond this scenario.

Validation for this checkpoint also includes the expanded Windows native installer suite (mapped
launcher replacement, interrupted publication repair, PATH ownership, application recovery), six
Windows update orchestration tests, CLI regression tests (102), targeted CLI/release type checks,
workflow lint and PowerShell/Bash syntax checks. The macOS harness uses a real published ICN and
requires a completion receipt; its corrected signed end-to-end run is still pending. Full-install
scripts and final cross-platform acceptance remain open. This checkpoint does not complete the goal.

### Phase 8 working results: shared macOS command registration

Desktop command linking and managed shell PATH registration now live in the shared privileged host
package. Desktop uses the same composed registration operation that the installation flow will use;
the shell directory constant is shared with link placement. Existing overwrite/removal policy is
preserved, and distribution applicability now covers these shared modules.

The moved tests pass (12), with the real fresh zsh/bash login and interactive shell test exercising
the shared registration entry twice before resolving and executing the bundled command. The Desktop
suite passes (210, two platform-gated skips); the other 12 formerly Desktop tests now run in the host
package. Targeted Desktop and daemon-management typechecks exit 0 with existing Effect diagnostics.
No installation script or fresh bundle publication is established by this extraction. Those remain
the next Phase 8 work, and this is not a separate checkpoint commit.

Signed macOS acceptance run 35956696719 remains in progress. Its native macOS, Windows and both
Linux jobs passed; the signed installer job still requires its completion receipt before acceptance.

### Phase 8 working results: first macOS bundle publication

The native transaction filesystem now supports same-volume, descriptor-relative initial publication
with atomic no-replacement rename. It validates the staged identity and private staging capability,
refuses existing destinations, and synchronizes both parents. The shared workspace exposes verified
first installation under its existing exclusive lease and revalidates admission at publication.
It verifies and synchronizes the staged bundle before mutation, reconciles identity after a reported
publication error, verifies the installed bundle, and completes parent synchronization. There is no
displaced bundle or exchange journal for initial publication; the atomic boundary leaves absence or
the complete bundle. Existing installations continue through the replacement transaction.

The native addon rebuilt successfully on macOS. Filesystem, recovery/transaction and workspace tests
pass (96), including fresh publication, identity mismatch, repeated publication refusal, preservation
of existing files/directories/symlinks, invalid staged verification, retained transaction refusal and
an injected error after successful publication without a second rename. The host package typecheck
exits 0. Installer command wiring, script bootstrap and signed fresh-install acceptance remain open;
these tests do not establish those end-to-end paths.

The archive installation operation now composes fresh publication and existing-bundle exchange
through the same workspace, staging and verifier services. It reconciles retained transactions before
an explicit retry, observes the existing installed version for replacement, and retires completed
transaction contents. It does not start an application owner. Integration exposed the native lease's
existing-bundle requirement; exclusive admission now permits an absent destination with current-user
lock ownership and confirms absence again after locking. Shared application admission still rejects
absence, and later owners reuse the same lock after publication.

The native addon rebuilt and the archive/prepared-installation, lease and workspace suites pass (26).
Coverage includes fresh archive installation, repeat replacement, no staging while a live shared
lease exists, and retention of lock identity from first installation into owner admission. The archive
test supplies controlled stager/verifier services; real signed archive and bootstrap acceptance remain
required. Targeted host typechecking exits 0. CLI entry and install scripts are still to be connected.

The hidden `_install-mac-application` CLI entry is now connected to archive installation and shared
command registration. It validates bounded input, requires an extracted executing bundle outside
the destination, verifies that source bundle, reads its sealed publisher configuration, and retains
application maintenance admission. It invokes no owner startup. Its seven input tests and seven
archive/prepared-installation tests pass; all 102 CLI regressions pass. Host and CLI targeted
typechecks exit 0. Shell bootstrap and signed invocation of this new command remain unverified.

Corrected signed macOS run 35956696719 completed with failure, not acceptance. Its finite update
printed installation success and native publisher acceptance passed, but foreground continuation
failed because inference exited with code 1 before readiness. The service also reported EBADF during
cleanup. The native Mac, Windows and both Linux jobs passed. Failure log is saved at
`/tmp/magnitude-headless-signed-mac-35956696719.log`; artifact download to
`/tmp/magnitude-headless-transfer/mac-35956696719` is in progress for diagnosis. There is no successful
foreground/desktop receipt from this run, and those acceptance requirements remain open.

The published macOS inference 0.1.5 installation was independently acquired into
`/tmp/magnitude-headless-transfer/published-mac-inference`. A bounded direct launch emitted Ready
and exited 0 on SIGTERM. The existing signed 0.0.504 fixture then served with that installation in
an isolated profile on port 11839, reported Headless Ready, answered a public `models status` query,
and exited 0 on SIGTERM. Its log is `mac-published-engine-serve.log` in the transfer directory.
This narrows the failed CI run but does not establish its cause or validate update continuation.
The ICN lifecycle now logs its already-bounded captured diagnostics on pre-readiness exit, while
keeping the ordinary error message concise. Six lifecycle tests pass and targeted ICN typechecking
exits 0. The CI artifact download remains active; no replacement run has been dispatched.

Full-installation release admission now reuses the existing signed release proof and shared channel
policy with an explicit caller-selected target. The small offer contains only the release and its
trusted-repository download URL; it can be projected from verified publisher records. Six tests pass
for fresh installation admission, architecture/package/OS mismatches, tampered content, missing trust,
foreign download origin, excess fields and channel selection. This is the metadata validation layer;
the installation scripts and publication wiring remain unfinished.

The Unix installer template now implements the macOS path: bounded HTTPS metadata/download,
size and digest checks, private bootstrap extraction, pinned Apple publisher verification and
Gatekeeper assessment, then the finite bundled installer. Its request includes the complete offer
and selected channel; the verified bundled command authenticates those against its publisher trust
before installation. A release-owned generator validates and pins metadata origin and Apple team.
The script does not launch an application owner. Twelve release metadata/generation tests and seven
installer-input tests pass; shell syntax and help pass, and release/host targeted typechecks exit 0.
The Linux branch, Windows script, generated-script publication and signed script acceptance remain
unfinished. The template is not yet a complete cross-platform installer and is not published.

The Unix template now has a Linux branch that selects apt/dnf, authenticates release metadata with
the pinned Ed25519 publisher key using Python 3 and OpenSSL, verifies package size/digest, then invokes
the package manager. It requires curl, Python 3 and an OpenSSL version supporting Ed25519. It preserves
the package manager's existing live-owner admission and never starts the application. Native Ubuntu
22 arm64 execution passed five maintained shell tests: valid input reached the recorded package
installation, while invalid signature, digest, architecture and channel did not. Only network and
privileged package mutation were substituted; JSON validation and OpenSSL verification were real.
Actual script-driven DEB/RPM installation and Windows bootstrap remain required. Release typechecking
and shell syntax pass. The signed macOS artifact download completed at the recorded transfer path.

The exact signed 0.0.503 artifact from run 35956696719 was extracted locally and passed strict deep
code-signature verification. It reached Headless Ready with published inference 0.1.5 on port 11839,
then exited 0 on SIGTERM; passive status confirmed Stopped/None. The isolated local log is
`/tmp/magnitude-headless-transfer/mac-35956696719-local.log`. This does not reproduce the CI failure
and does not establish the original update-continuation receipt.

The Windows PowerShell bootstrap now downloads and authenticates a standalone CLI verifier before
using its embedded publisher key to authenticate the installation offer and exact installer bytes.
It also verifies installer Authenticode publisher/timestamp, waits for `/S` setup and updates the
current shell PATH. Windows CLI release compilation now includes its existing signing pipeline.
Six verifier tests pass for valid input, bad signature, modified/truncated/oversized bytes and wrong
target. Seven script-generation tests pass, CLI targeted typechecking exits 0, and native Windows
PowerShell parses the template without errors. Actual signed bootstrap execution, acceptance-key
build configuration, package/publication integration and complete script regression lanes remain open.

Release tooling now prepares a fresh static installation distribution from verified publisher records:
Unix/PowerShell scripts plus channel/target offers. Mixed versions, duplicate targets and invalid
signatures fail before output creation; an existing hosting directory is never overwritten. The
publisher can export its accepted records through `MAGNITUDE_INSTALL_PUBLICATIONS_OUTPUT`, and
`prepare-installation-distribution.ts` consumes those records with explicit hosting/signing inputs.
This prepares local output only; it does not deploy script URLs or move a live channel. Ten script and
distribution tests pass, and release typechecking exits 0. Windows download cancellation now covers
stream reads as well as headers; the updated template parses in native Windows PowerShell.
Real HTTPS/script/package acceptance and acceptance-key compilation are still open.

The maintained Linux HTTPS/script acceptance passed on the disposable Ubuntu 22 arm64 VM using the
existing compiled 0.1.9-1 DEB fixture. The prior idle test package was removed first. A temporary local
TLS server hosted generated metadata and the real package; curl used an isolated test CA and explicit
GitHub-to-loopback mapping, with no global trust changes. The generated installer authenticated the
release, downloaded and verified the package, and installed through real apt. Passive status remained
Stopped. The installed public command reached Headless Ready, answered model status and hardware,
exited 0 on SIGTERM, and supported a repeat script installation while remaining stopped. The test
server and owner were retired; the package remains installed for later regression tests.

Receipt and logs are under `/tmp/magnitude-headless-transfer/ubuntu-script-evidence`, copied from
VM `/tmp/magnitude-script-acceptance.IHDYoGDA`. This proves script/package integration against that
fixture, not a newly rebuilt final cohort. Fedora script execution, Windows/Mac signed script
acceptance, fault cases and the final application regression lanes remain open. Sixteen release
metadata/script-generation tests pass.

The same HTTPS/script acceptance passed on Fedora 44 arm64 through real dnf with the existing
0.1.9-9 RPM fixture. The prepared hosting tree was generated on Ubuntu, then served and consumed
inside Fedora with an isolated TLS trust file. Initial Stopped, Headless Ready, model/hardware
queries, SIGTERM exit 0, repeat installation and final Stopped all passed. Local evidence is
`/tmp/magnitude-headless-transfer/fedora-script-evidence`, copied from Fedora
`/tmp/magnitude-script-acceptance.suzquiMJ`. The test package remains installed; no test owner or
HTTPS server remains. Both DEB and RPM script results still use prior compiled fixture cohorts.

Acceptance builds can now embed their isolated bootstrap publisher key at compile time; normal
builds retain the checked-in production public key. The maintained compiled verifier acceptance
passed on this Mac and in Windows: real CLI compilation, accepted signed fixture bytes, rejected
same-length tampering, ignored runtime key override, and no application state creation. Test paths
include spaces and an apostrophe. This does not establish Authenticode/bootstrap script execution;
that remains the next Windows gate. Release and CLI typechecks exit 0.

The signed Windows bootstrap acceptance now passes against a freshly built 0.0.505 fixture from
Phase 8 source. Its application and standalone verifier used an isolated test publisher with real
Authenticode timestamps. The generated PowerShell script was fetched over HTTPS and downloaded the
verifier and installer through their normal HTTPS URLs, served locally inside the disposable VM.
The script accepted both executable signatures, authenticated release metadata and installer bytes,
completed native silent setup, and exposed the registered command in the invoking shell. Version was
0.0.505 and passive status was Stopped/None. Repeat installation passed. A modified release digest
was rejected with application release verification failure, leaving 0.0.505 stopped and unchanged.

This test caught an unnecessary ARM Windows rejection in the script; removing that restriction lets
Windows execute the existing x64 release through its supported emulation. The VM acceptance used
that actual x64 package. Seven generation tests and release targeted typechecking pass. The VM's
ordinary script-file execution policy was preserved; the downloaded script ran as a script block,
as with the intended interactive bootstrap invocation.

Evidence is `/tmp/magnitude-headless-transfer/windows-script-evidence.txt`; full guest build and
script logs are in `C:\Users\trg\hs-phase8`. The temporary HTTPS server was stopped, original hosts
bytes restored, and both temporary root certificates and private certificates removed. Cleanup was
independently checked. The 0.0.505 application remains installed and stopped. This establishes signed
script replacement/repeat/failure behavior, not a fresh-machine install or foreground serve acceptance
for this new cohort. Those checks, maintained orchestration of this HTTPS lane, signed Mac bootstrap,
and final desktop/CLI/remote regressions remain open. No Phase 8 checkpoint has been committed yet.

The maintained installed-headless acceptance passed against the script-installed Windows 0.0.505
application: initially stopped, public launcher `serve` reached Headless Ready with published ICN
0.1.5, model status and hardware queries succeeded, application-control Quit returned launcher exit 0,
and final status was Stopped/None. Native process inspection found no remaining Magnitude processes.
Local receipt: `/tmp/magnitude-headless-transfer/windows-script-serve-evidence.txt`; guest details:
`C:\Users\trg\hs-phase8\serve-acceptance`. This closes the Windows script-to-serve/query integration
check for that cohort, without claiming model inference or desktop UI coverage.

The first attempt used an absent profile with a separately located state directory and exposed a
startup initialization defect: native update-directory creation requires its parent to exist.
Shared Windows recovery now creates the profile parent before native private-cache creation. This
preserves existing-cache validation and repair rules. Five real Windows staging tests pass, including
the new absent-profile case; six foreground-update orchestration tests also pass. The installed
0.0.505 fixture predates this small fix: its successful packaged run used the ordinary shared
profile/state layout. A subsequent packaged cohort must cover the separate-root case as well.

The signed Mac acceptance is now wired to install its first signed fixture through the generated
HTTPS shell script instead of direct ZIP extraction. It checks fresh and repeat installation,
signature rejection, isolated shell registration, stopped state and native publisher/notarization
acceptance, then runs the maintained installed serve/query test before its existing update sequence.
CI retains shell and serve receipts/logs. This new signed lane has not run yet. Shell syntax passes;
release and host targeted typechecks pass, and 138 affected Mac/native tests pass. Signed Mac
execution and remaining script fault/regression gates still precede the Phase 8 completion claim.

Checkpoint preparation: the full ordinary CLI suite passes 102 tests and the Desktop suite passes
210 tests with two platform skips after shared command-registration extraction. Sixteen release
metadata/distribution/script tests pass. CLI targeted typechecking passes. The Phase 8 implementation
is substantial enough to checkpoint for signed Mac execution; this checkpoint does not close Phase 8.
The Windows and Linux script integrations have real native evidence, while signed Mac script execution,
further failure/lease cases and final fresh-profile packaged coverage remain outstanding.
