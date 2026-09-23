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
