# Shared application updates: source review and implementation plan

Status: proposal, not implemented. Reviewed 2026-09-23 against Magnitude `772cfacb`.
Companion to `/Users/trg/Downloads/headless-serve.md`; does not implement that document.

## Agreed behavior

- The installed desktop, CLI and ACN remain one release and update together.
- Desktop and headless owners share preparation, storage and installation policy.
- `serve` checks and downloads according to the existing preference while running. A ready update
  reports: “Update ready. Stop the server and run `magnitude serve` again to install.”
- An automatic update never stops a live server. Normal shutdown only stops it.
- On the next owner startup, apply an already-prepared update before spawning ACN, then continue the
  original startup intent. No prepared update means no network dependency in startup.
- An explicit desktop restart-to-update remains possible. Headless users stop and start `serve`;
  service-manager users restart their configured unit. Do not invent a service-management command.
- Failed attempts require explicit retry; a healthy existing installation may still serve. An
  uncertain or inconsistent installation must be repaired before service startup.
- Startup requiring unavailable authorization defers installation and explains the explicit action.
  Linux system-package installation still requires root; this constraint cannot be abstracted away.

## Recommendation and evidence

Implement a narrow macOS bundle-installation transaction in daemon-management. Reuse the existing
Windows NSIS transaction and Linux package-manager installation. Extract shared Effect orchestration
from desktop. Do not implement a general-purpose update framework.

Sparkle is a viable alternative. Its standard integration is not inherently difficult. The additional
work for Magnitude is reconciling its appcast/signature pipeline, application termination/relaunch,
helper lifecycle and authorization with our existing signed release protocol and two owner kinds.
Those integration costs are an inference from the source, not a measured prototype. A native
prototype is still necessary before claiming our custom approach is smaller overall.

Sources were cloned read-only outside the workspace and inspected at these exact revisions:

### Sparkle `fd34238cbbc5db4a6e8343c62ae4dea939b06ea4`

- [SUPlainInstaller.m](https://github.com/sparkle-project/Sparkle/blob/fd34238cbbc5db4a6e8343c62ae4dea939b06ea4/Autoupdate/SUPlainInstaller.m): stages on the destination volume when necessary, attempts atomic exchange,
  falls back to sequential moves, handles quarantine and ownership, and conditions exchange on
  signing identity/custom update security policy. Adopt destination-volume preparation and the
  signing constraints; do not silently inherit all fallback behavior.
- [SUFileManager.m](https://github.com/sparkle-project/Sparkle/blob/fd34238cbbc5db4a6e8343c62ae4dea939b06ea4/Sparkle/SUFileManager.m): implements exchange with `RENAME_SWAP`; separate rename paths avoid following
  symlinks, including descriptor-relative operations. Our native boundary must protect path identity.
- [SUUpdateValidator.m](https://github.com/sparkle-project/Sparkle/blob/fd34238cbbc5db4a6e8343c62ae4dea939b06ea4/Sparkle/SUUpdateValidator.m): distinguishes archive authentication from extracted-code validation and has
  extensive key-transition policy. Keep both validation stages, reuse our release trust policy.
- [AppInstaller.m](https://github.com/sparkle-project/Sparkle/blob/fd34238cbbc5db4a6e8343c62ae4dea939b06ea4/Autoupdate/AppInstaller.m): coordinates extraction, validation, staged installation, agent termination and
  relaunch. This application-oriented lifecycle is additional integration, not a bare bundle-swap API.
- Also inspected `Tests/SUFileManagerTest.swift`, installer tests and the framework license.

### Squirrel.Mac `5c9e2133c09d6f8e2e3c5a45c5b0ffc00448c58a`

- [SQRLInstaller.m](https://github.com/Squirrel/Squirrel.Mac/blob/5c9e2133c09d6f8e2e3c5a45c5b0ffc00448c58a/Squirrel/SQRLInstaller.m): privately prepares the update, validates its signature, persists ownership of the
  displaced bundle, and restores it after interruption. Adopt durable recovery before mutation.
- [SQRLCodeSignature.m](https://github.com/Squirrel/Squirrel.Mac/blob/5c9e2133c09d6f8e2e3c5a45c5b0ffc00448c58a/Squirrel/SQRLCodeSignature.m): checks the designated requirement through Security.framework, including nested
  code, strict validation and all architectures. Checking only a displayed Team ID is insufficient.
- [ShipIt-main.m](https://github.com/Squirrel/Squirrel.Mac/blob/5c9e2133c09d6f8e2e3c5a45c5b0ffc00448c58a/Squirrel/ShipIt-main.m): restores previous state and bounds repeated installation attempts. Magnitude should
  retain its stricter existing “failed attempt requires explicit retry” behavior.
- `SQRLUpdater.m`, `SQRLTerminationListener.m` and `SQRLShipItLauncher.m` use running-application
  identity, application termination observations and launchd jobs. Linking the framework without
  Electron does not automatically make these assumptions fit a foreground headless owner.

The frameworks' permissive licenses require preserving applicable notices if implementation code
is reused. This plan borrows behaviors, not source fragments.

## Scope of the custom macOS backend

Support Magnitude's complete signed/stapled ZIP, fixed bundle identity, supported architecture and
release channel, macOS 13+, and a local installation with a supported atomic-exchange filesystem.
The target is the resolved installation, not an arbitrary path supplied by an untrusted request.
The installer/helper is signed with the app's publisher identity.

No deltas, appcasts, package installation on macOS, arbitrary third-party bundles, updater UI,
automatic process killing, permanent privileged service, or general certificate-migration engine.
Do not weaken archive authentication to implement a signing fallback. An unsupported volume,
read-only location, unfamiliar update security policy or missing permission yields a deferred or
failed update with the old installation preserved. Supporting broader installation permissions is
a separate capability; startup must not silently prompt or change installation ownership.

Apple documents atomic exchange using `renameatx_np(..., RENAME_SWAP)` on supported filesystems:
[rename manual source](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/rename.2).
Atomic namespace exchange is not by itself a complete durability or recovery protocol.

## Shared responsibilities

1. **Update preparation** owns discovery, channel selection, download, verification, status, preference,
   and prepared artifacts. Preserve `ApplicationUpdate`'s admission and cancellation behavior.
2. **Installation transaction** owns exclusion, native replacement and reconciliation. Inputs describe
   the verified release and installation identity; output distinguishes Installed, Deferred, Failed,
   and RepairRequired. A spawned helper is not successful installation.
3. **Startup continuation** belongs to the invoking owner: desktop visibility intent or foreground
   headless execution. Installation must not hard-code launching Electron.

Use Effect services, tagged schemas for durable state, branded transaction identities, and scoped
native capabilities. Prepared artifact state and filesystem transaction state are distinct:
`Attempted` alone cannot tell which bundle occupies the installed path after a crash.

Move reusable orchestration from `desktop/src/application-update.ts`, `hosted-update-source.ts`,
`prepared-update-installation.ts`, scheduling and identity into daemon-management. Keep release
contracts/trust in release, portable observation types in SDK, and window/tray/resume wiring in
desktop. CLI commands consume owner control; when no owner runs, explicit maintenance can acquire
ownership for a finite install. Passive commands do not trigger updates.

## macOS transaction

1. Acquire the installation lease while application ownership excludes an existing service. Confirm
   no preexisting native updater is active. Establish a lock order that allows handoff without waiting
   for application ownership while preventing that owner from exiting. Contenders release and refuse
   when installation is active; they must not deadlock an installer needing application ownership.
2. Resolve and retain native installation-parent identity. Validate the target bundle and allowed
   publisher; bind the transaction to this exact installation. Multi-user installations require
   installation-wide exclusion; a per-user application lock alone does not prove another user's
   owner is absent. Either provide that admission or explicitly limit automatic replacement to an
   installation owned exclusively by this user.
3. Authenticate the retained archive again. Create a private transaction directory on the target
   volume. Use a vetted ZIP extraction path preserving framework links and executable bits; validate
   entry containment, link targets and expected bundle layout. Do not write into the live bundle.
   Verify the actual staged bytes after any cross-volume copy, not only an earlier source copy.
4. Validate nested code through Security.framework against trusted publisher/designated requirements,
   expected bundle ID, version and architecture. Preserve the notarization chain. Test quarantine and
   Gatekeeper behavior with signed packaged builds; do not indiscriminately clear security attributes.
5. Persist transaction ID, installed/staged locations and their expected identities, old/new release
   identities and an exchange-intent state. Sync staged content and transaction metadata before
   publishing mutation authority. Resolve the exact filesystem durability sequence in the native
   prototype; do not substitute a JSON write for durable preparation.
6. Under retained exclusion, exchange staged and installed bundles using descriptor-relative native
   operations. Refuse unsupported exchange rather than introducing a path-missing two-rename fallback.
   Never repeat an exchange just because completion was not recorded: that could reinstall the old app.
7. Validate the target's resulting identity, sync and record installation completion. Keep the old
   bundle until commit is durable. Cleanup is repeatable and limited to transaction-owned entries;
   cleanup failure does not undo a committed update or authorize deleting unknown files.
8. Continue startup using the new executable. Recover before spawning ACN. Do not roll back binaries
   automatically merely because subsequent application/model startup fails: data migrations can make
   that unsafe. Filesystem rollback is limited to the uncommitted installation transaction.

### Recovery decisions

| Observed state under exclusive admission | Action |
| --- | --- |
| Installed = old; staging = new; no exchange occurred | Preserve old; mark interrupted attempt for explicit retry |
| Installed = new; displaced = old; completion record missing | Validate identities and complete commit; do not exchange again |
| Installed = new; committed | Retry cleanup only |
| Invalid replacement; verified transaction-owned old bundle available | Restore old before declaring recoverable failure; record failure |
| Missing/unknown identities or failed restoration | Preserve evidence, report RepairRequired, do not start ACN |

The journal records intent; actual validated filesystem identities determine which mutation happened.
Initial installation into an absent target uses a separate no-overwrite publication path under the
same verification/admission. The install script must not implement its own shell bundle-swap logic.

## Foreground startup continuation

No server is live yet, but the initiating process must remain correctly tracked by the terminal or
service manager. “Spawn detached replacement and exit” does not satisfy foreground `serve`.

**macOS:** prototype an exec continuation through a verified helper outside the replaced bundle,
then exec the new CLI with the same foreground identity and invocation. Exclusion must explicitly
survive the transition: existing native locks are close-on-exec. Introduce narrowly typed adoption
of an inherited update capability if needed, not general inheritance of owner locks into ACN.
Validate cancellation at every phase; cancellation may prevent serving but cannot abandon a
half-completed transaction or launch a background server.

**Windows:** first validate a waiting original process with a copied installer helper. Current NSIS
replacement renames old/new trees, and `FinishReplacement` permits retaining old files when cleanup
fails. This makes deferred cleanup while the old CLI image remains loaded plausible, not proven.
The old process must release application ownership/installation-directory handles, use a working
directory outside the target, stop reading resources from the old path, and supervise the new owner
after installation. Its retained lifetime must propagate console cancellation, exit status and
service-manager termination; native jobs and process handles must preserve descendant cleanup.
Test repeated updates while an earlier CLI image remains mapped. Do not assume Windows cannot
rename a loaded installation, and do not promise it can based only on source inspection.

If this fails native acceptance, a launcher outside the replaceable payload is the fallback. That
changes PATH registration/packaging and needs a minimal stable launcher protocol and its own update
policy. It is an explicit architectural decision, not something to hide inside the macOS installer.
Do not declare seamless Windows startup updates complete until this gate is passed.

**Linux:** release shared installation admission only within installer exclusion, run the existing
package transaction through explicit root authorization, then reacquire admission before execution.
Noninteractive startup without authorization serves the intact installed version and reports a
deferred update. OS-managed upgrades must obey the same admission. Run the service as the original
user, not root. Test systemd cleanup: detaching a helper does not move it out of the unit's cgroup.

## Windows defect and release order

The reported private-directory failure is confirmed by source inspection. It is independent of
Squirrel/Sparkle and should be fixed before the broader extraction. See the
[bug investigation](../../bugs/26-09-23/windows-update-directory.md).

1. Fix creation plus existing bad-directory recovery; repair the native end-to-end acceptance path.
2. Ship a fixed installer and communicate manual installation for affected older clients. Test that
   the manually repaired installation can subsequently update through the normal updater.
3. Extract shared orchestration without changing desktop behavior; retain existing tests.
4. Implement and validate the macOS native transaction, including failure injection and signed builds.
5. Pass foreground startup-continuation gates on every supported platform; integrate `serve`.
6. Remove Electron macOS updater wiring once the custom backend passes acceptance. Exclude/drain a
   still-active old ShipIt operation during the transition; do not run both installers concurrently.

Update applicable design documents with implementation, including release updates/distribution,
native ownership, and CLI lifecycle. This proposal does not silently replace their current contracts.

## Acceptance gates

- Real signed macOS x64/arm64 builds: nested signature failures, wrong publisher/ID/version/arch,
  invalid ZIP entries/links, valid framework links, cross-volume download cache, unsupported exchange,
  read-only target, insufficient space, and Gatekeeper behavior on supported OS versions.
- Inject process termination before/after every durable write and exchange; restart recovery must
  distinguish old/new identities without toggling versions. Test sync failures separately from process
  interruption; VM/storage-level power-loss validation is stronger than killing a process.
- Race desktop, `serve`, explicit installer and a second startup, including multiple users if supported.
  No installer mutates underneath an admitted live server, and no contender starts mid-install.
- Confirm no automatic server shutdown on download completion; next startup installs before ACN;
  network outage without prepared bytes does not block startup.
- Preserve desktop visibility, foreground console, signals, exit code, working directory and supported
  profile selection. Ctrl+C during update must not leave a detached serving process.
- Native Windows fresh and poisoned profiles: real download through Ready and install, real ACL adapter,
  actual NSIS replacement, Authenticode and loaded CLI/addon files, console/SSH/service-manager runs,
  repeated upgrades and failure recovery. Mocks/cross-compilation are not sufficient.
- Linux authorization decline/unavailability, package repair state, shared-lease exclusion and no
  accidental privileged service startup.

Research limits: no native Windows reproduction or signed macOS replacement was executed in this
investigation. Source-derived feasibility is distinguished above from behavior requiring execution.
