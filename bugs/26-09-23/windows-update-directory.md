# Windows update staging rejects its own download directory

Status: repair implemented on `headless`; native and staging tests passed in the Windows VM.
Full signed packaged upgrade acceptance remains pending. Original investigation source: `772cfacb`.

## Original failure

`desktop/src/main.ts` supplies `<data>/updates` as `cacheDirectory` to the hosted download source.
`desktop/src/hosted-update-source.ts` creates that path with ordinary `FileSystem.makeDirectory`
and `mode: 0o700`, then creates a scoped transfer directory beneath it. This does not request
the explicit protected Windows ACL required by our private storage contract.

After verification, `desktop/src/windows-update-source.ts` calls `PreparedUpdateStore.prepare`.
`packages/daemon-management/src/desktop-native/prepared-update.ts` calls
`PrivateFilePermissions.prepareDirectory` on the same `<data>/updates` directory. The Windows
implementation calls `magnitude_prepare_private_directory` in `native/windows-security.c`.
Creation with its private descriptor does not replace an existing directory's permissions. The
subsequent validation requires current-user ownership, a protected DACL, exactly one current-user
full-access ACE, exact inheritance flags, and a real non-reparse directory. A normal inherited
directory fails. The resulting error becomes “The downloaded update could not be saved.”

The user's VM report of inherited SYSTEM/Administrators/user ACEs is consistent with this exact
failure. Native security tests deliberately assert refusal to repair broad directories.

## Scope and corrections to the original notes

- The problematic mkdir is on **download**, not every metadata check. Existing directories are
  reused; mkdir does not reset an already-correct protected ACL. Normal fresh profiles trigger it;
  an already correctly secured directory is an exception to “every Windows user.”
- The offending lines exist in commit `183a1cc1`, the local `@magnitudedev/cli@0.1.0` and `0.1.4`
  source tags, and current main after the 0.1.5 release preparation. This establishes source presence,
  not independent verification of the exact signed executables shipped to every user.
- Deleting the directory before retrying the old updater recreates the problem.
- Unix private-directory preparation runs mkdir plus chmod and does not have this ACL mismatch.
- A server release cannot change the old client's local creation/validation path. Normal in-app
  installation of a fixed release is blocked on affected clients.
- Manual installation of **0.1.5** upgrades the application but leaves the same updater defect.
  A future fixed installer is the recovery target; no specific future version has been allocated here.
- Reinstalling app binaries alone does not repair `<data>/updates`. The fixed client must handle the
  bad directory left by prior attempts, or recovery instructions must explicitly address it.

## Repair requirements

Treat transfer scratch and trusted prepared-update storage as distinct responsibilities.

1. Move ephemeral download scratch outside the prepared-update directory. Authenticate bytes before
   publication; the store alone creates its private root through `PrivateFilePermissions`.
   Alternatively, create the existing root with the native private capability before download;
   either approach alone fixes only fresh profiles.
2. Add narrowly scoped recovery for the known old cache under exclusive application/update admission.
   Prefer abandoning the old cache as an untrusted artifact: retain native handles, require the
   expected current-user-owned local non-reparse directory and known cache layout, move it to a
   unique quarantine sibling, and create a fresh private root. Do not import an old prepared record
   or executable as trusted; redownload and verify. Unknown entries/ownership/reparse state fail
   with actionable repair guidance, without recursive deletion or automatic permission rewriting.
   Make interruption between retirement and creation harmless and cleanup repeatable. This recovery
   is limited to the reported cache defect, not a compatibility mode for unsafe private storage.
3. Keep the general private-directory validator strict. Do not solve this by accepting inherited
   ACLs globally or by recursively changing permissions on arbitrary existing paths.
4. Preserve detailed diagnostic causes while showing a useful directory/recovery error in the UI.

The exact native recovery operation needs Windows acceptance, including paths with junction ancestors
and rename races. Do not implement it as unchecked JavaScript `exists` + recursive deletion.

## Original test gaps

- `desktop/src/windows-update-source.test.ts` substitutes ordinary mkdir/write and a no-op protection
  function for native private permissions, creates its archive separately, and directly calls stage.
  It never exercises the offending download-to-native-stage sequence.
- `native/windows-security-test.c` tests strict ACL behavior correctly, independently of downloading.
- `desktop/src/fixtures/windows-hosted-update.mjs` still reads `updates/preferences.json` and
  `updates/installation-key.pem`; current code uses the canonical config and root `identity.pem`.
  Update those assertions. Source inspection does not establish whether/how recently this fixture ran.

## Required verification and rollout

- A fresh disposable Windows profile with a normally inherited `.magnitude` root must reach Ready
  through actual hosted download, Authenticode verification and the real private-file adapter.
- A profile poisoned by a released old client must recover safely after installing the fixed build.
- Refuse wrong owner, unexpected contents, reparse points and malformed recovery state; preserve data.
- Verify cancellation/retry and interruption during cache recovery.
- Complete a signed packaged update from the fixed build to a second build, including installer exit,
  relaunch, version change and persisted user state. Then repeat to detect retained-state regressions.
- Communicate direct installation of the fixed release for affected old versions. Do not promise that
  publishing the fix repairs old binaries automatically. A separately validated manual ACL repair
  could enable an old updater, but is not a server-side remedy or the proposed default user workflow.

## Implemented checkpoint

The transfer source now derives separate scratch storage from the profile root. Windows updater
bootstrap runs narrow native cache recovery before reading prepared state. Recovery recognizes only
the original inherited user/administrator/system ACL, retains the directory handle through rename,
preserves recognized contents under an identity-derived sibling, and creates new private storage.
Generic private-directory validation remains unchanged. Refusal reaches the updater UI as a
specific directory error rather than the generic setup error.

Native tests passed under the VM's ordinary user, including inherited-cache recovery, explicit/broad
ACL refusal, unknown-entry preservation, repeat recovery, and root/child junction refusal under both
Node and Bun. Staging tests now use real native private permissions and cover fresh and inherited
profiles, corrupt bytes, and publisher-verification failure. Transfer separation and interrupted
transfer cleanup have regression tests. The hosted acceptance fixture uses current configuration
paths and starts with an inherited cache to exercise upgrade recovery.

This checkpoint has not published a release or completed the signed hosted upgrade/rollout gates.
See the [checkpoint ledger](../../specs/26-09-23/headless-checkpoints.md) for executed evidence and
remaining acceptance work.
