import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { basename, dirname, join, resolve } from "node:path"
import { exchangeMacUpdate, publishMacInstallation, type MacUpdateVersions } from "./mac-update-transaction"
import type { MacBundleExpectation } from "./mac-update-validation"
import { MacUpdateAdmission, type MacExclusiveInstallationLease } from "./mac-update-lease"
import { MacUpdateFilesystem, MacUpdateFilesystemFailed } from "./mac-update-filesystem"
import { MacUpdateRepairRequired, recoverMacUpdateTransaction, retireMacUpdateBundle } from "./mac-update-recovery"

export class MacUpdateInstallationBusy extends Schema.TaggedError<MacUpdateInstallationBusy>()("MacUpdateInstallationBusy", {}) {
  override get message() { return "Magnitude is using this installation. Stop the running application before installing an update." }
}

/** Stable discovery is installation-relative; every access retains exclusive installation admission. */
export const openMacUpdateWorkspace = (bundle: string, create: boolean, retained?: MacExclusiveInstallationLease) => Effect.gen(function* () {
  if (resolve(bundle) !== bundle || !bundle.endsWith(".app") || bundle.includes("\0")) return yield* new MacUpdateRepairRequired()
  const admission = yield* MacUpdateAdmission
  if (retained && retained.bundle !== bundle) return yield* new MacUpdateRepairRequired()
  const lease = retained ? Option.some(retained) : yield* admission.exclusive(bundle)
  if (Option.isNone(lease)) return yield* new MacUpdateInstallationBusy()
  yield* lease.value.validate
  const fs = yield* FileSystem.FileSystem
  const native = yield* MacUpdateFilesystem
  const installedName = basename(bundle)
  const installed = yield* native.open(dirname(bundle), false)
  const stagingName = `.${installedName}.update`
  const stagingPath = join(installed.path, stagingName)
  const existing = yield* native.inspect(installed, stagingName)
  if (Option.isNone(existing)) {
    if (!create) return Option.none()
    // Do not accept an existing entry here: a competing creator did not hold our installation lease.
    yield* fs.makeDirectory(stagingPath, { mode: 0o700 })
  }
  const staging = yield* native.open(stagingPath, true)
  const validate = Effect.gen(function* () {
    yield* lease.value.validate
    if (!Option.contains(yield* native.inspect(installed, stagingName), staging.identity)) return yield* new MacUpdateRepairRequired()
  })
  yield* validate
  yield* Effect.addFinalizer(() => validate.pipe(
    Effect.zipRight(native.removeEmptyDirectory(installed, stagingName, staging)), Effect.ignore))
  // Persist the private directory's discovery entry before extraction or journal publication.
  yield* native.sync(installed)
  const protectedFilesystem = MacUpdateFilesystem.of({ ...native,
    exchange: (...args) => validate.pipe(Effect.mapError(() => new MacUpdateFilesystemFailed()), Effect.zipRight(native.exchange(...args))),
    publish: (...args) => validate.pipe(Effect.mapError(() => new MacUpdateFilesystemFailed()), Effect.zipRight(native.publish(...args))),
  })
  const recover = validate.pipe(Effect.zipRight(recoverMacUpdateTransaction(installed, installedName, staging)),
    Effect.provideService(MacUpdateFilesystem, protectedFilesystem))
  const retire = validate.pipe(Effect.zipRight(retireMacUpdateBundle(installed, installedName, staging)),
    Effect.provideService(MacUpdateFilesystem, protectedFilesystem))
  const exchange = (versions: MacUpdateVersions) => validate.pipe(
    Effect.zipRight(exchangeMacUpdate(installed, installedName, staging, versions)),
    Effect.provideService(MacUpdateFilesystem, protectedFilesystem))
  const clearUnpublishedStaging = Effect.gen(function* () {
    yield* validate
    if (Option.isSome(yield* native.readRecord(staging))) return yield* new MacUpdateRepairRequired()
    const candidate = yield* native.inspect(staging, "Magnitude.app")
    if (Option.isSome(candidate)) yield* native.removeTree(staging, "Magnitude.app", candidate.value)
    yield* native.sync(staging)
  })
  const publish = (expected: MacBundleExpectation) => validate.pipe(
    Effect.zipRight(publishMacInstallation(installed, installedName, staging, expected)),
    Effect.provideService(MacUpdateFilesystem, protectedFilesystem))
  return Option.some({ installed, installedName, staging, validate, recover, retire, exchange, publish, clearUnpublishedStaging })
}).pipe(Effect.mapError(error => error._tag === "MacUpdateInstallationBusy" ? error : new MacUpdateRepairRequired()))
