import { Effect, Option, Schema } from "effect"
import { randomUUID } from "node:crypto"
import { join } from "node:path"
import { MacUpdateFilesystem, type MacUpdateDirectory } from "./mac-update-filesystem"
import { MacBundleVerifier, type MacBundleExpectation } from "./mac-update-validation"
import { ExchangeIntent, MacUpdateJournal, MacUpdateRepairRequired, recoverMacUpdateTransaction, type MacUpdateRecoveryResult } from "./mac-update-recovery"

export const MacUpdateVersions = Schema.Struct({
  previous: Schema.NonEmptyString.pipe(Schema.maxLength(96)),
  replacement: Schema.NonEmptyString.pipe(Schema.maxLength(96)),
  architecture: Schema.Literal("arm64", "x64"),
})
export type MacUpdateVersions = typeof MacUpdateVersions.Type
export class MacUpdatePreparationFailed extends Schema.TaggedError<MacUpdatePreparationFailed>()("MacUpdatePreparationFailed", {}) {
  override get message() { return "The application update could not be prepared for installation." }
}

/** First installation has no displaced bundle; atomic publication leaves either absence or the complete bundle. */
export const publishMacInstallation = (installed: MacUpdateDirectory, installedName: string,
  staging: MacUpdateDirectory, expected: MacBundleExpectation) => Effect.uninterruptibleMask(restore => Effect.gen(function* () {
  const fs = yield* MacUpdateFilesystem
  const verifier = yield* MacBundleVerifier
  const replacement = yield* restore(Effect.gen(function* () {
    if (Option.isSome(yield* fs.readRecord(staging)) || Option.isSome(yield* fs.inspect(installed, installedName))) {
      return yield* new MacUpdatePreparationFailed()
    }
    const candidate = yield* fs.inspect(staging, "Magnitude.app")
    if (Option.isNone(candidate)) return yield* new MacUpdatePreparationFailed()
    yield* verifier.verify(join(staging.path, "Magnitude.app"), expected)
    yield* fs.syncTree(staging, "Magnitude.app", candidate.value)
    return candidate.value
  }).pipe(Effect.mapError(() => new MacUpdatePreparationFailed())))
  yield* fs.publish(installed, installedName, staging, "Magnitude.app", replacement).pipe(
    Effect.catchTag("MacUpdateFilesystemFailed", () => Effect.void))
  // A reported sync error can follow successful rename. Reconcile without repeating publication.
  if (!Option.contains(yield* fs.inspect(installed, installedName), replacement) ||
      Option.isSome(yield* fs.inspect(staging, "Magnitude.app"))) return yield* new MacUpdatePreparationFailed()
  yield* verifier.verify(join(installed.path, installedName), expected)
  yield* fs.sync(installed)
  yield* fs.sync(staging)
  return { _tag: "Installed", version: expected.version } as const
})).pipe(Effect.mapError(() => new MacUpdatePreparationFailed()))

/** The finite installer owns exclusion and has authenticated/extracted the retained archive. */
export const exchangeMacUpdate = (installed: MacUpdateDirectory, installedName: string, staging: MacUpdateDirectory,
  versions: MacUpdateVersions): Effect.Effect<MacUpdateRecoveryResult, MacUpdatePreparationFailed | MacUpdateRepairRequired, MacUpdateFilesystem | MacBundleVerifier> =>
  Effect.uninterruptibleMask(restore => Effect.gen(function* () {
    const fs = yield* MacUpdateFilesystem
    const verifier = yield* MacBundleVerifier
    const record = yield* restore(Effect.gen(function* () {
      if (Option.isSome(yield* fs.readRecord(staging))) return yield* new MacUpdatePreparationFailed()
      if (installed.identity.split(":")[0] !== staging.identity.split(":")[0]) return yield* new MacUpdatePreparationFailed()
      const previous = yield* fs.inspect(installed, installedName).pipe(Effect.mapError(() => new MacUpdateRepairRequired()))
      const replacement = yield* fs.inspect(staging, "Magnitude.app")
      if (Option.isNone(previous)) return yield* new MacUpdateRepairRequired()
      if (Option.isNone(replacement)) return yield* new MacUpdatePreparationFailed()
      const intent = yield* Schema.decodeUnknown(ExchangeIntent)({ _tag: "ExchangeIntent", protocol: 1, transaction: {
        id: yield* Effect.sync(randomUUID), installedParent: installed.identity, stagingParent: staging.identity, installedName,
        architecture: versions.architecture, previous: { identity: previous.value, version: versions.previous },
        replacement: { identity: replacement.value, version: versions.replacement },
      } })
      yield* verifier.verify(join(installed.path, installedName), { version: versions.previous, architecture: versions.architecture }).pipe(
        Effect.mapError(() => new MacUpdateRepairRequired()))
      yield* verifier.verify(join(staging.path, "Magnitude.app"), { version: versions.replacement, architecture: versions.architecture })
      yield* fs.syncTree(staging, "Magnitude.app", replacement.value)
      if (!Option.contains(yield* fs.inspect(installed, installedName).pipe(Effect.mapError(() => new MacUpdateRepairRequired())), previous.value)) {
        return yield* new MacUpdateRepairRequired()
      }
      if (!Option.contains(yield* fs.inspect(staging, "Magnitude.app"), replacement.value)) return yield* new MacUpdatePreparationFailed()
      yield* fs.sync(installed)
      yield* fs.sync(staging)
      return intent
    }).pipe(Effect.mapError(error => error._tag === "MacUpdateRepairRequired" ? error : new MacUpdatePreparationFailed())))
    // Once publication starts, finish reconciliation before honoring cooperative cancellation.
    const text = yield* Schema.encode(Schema.parseJson(MacUpdateJournal))(record).pipe(Effect.mapError(() => new MacUpdatePreparationFailed()))
    yield* fs.writeRecord(staging, Buffer.from(text)).pipe(Effect.mapError(() => new MacUpdatePreparationFailed()))
    yield* fs.exchange(installed, installedName, record.transaction.previous.identity,
      staging, "Magnitude.app", record.transaction.replacement.identity).pipe(Effect.catchTag("MacUpdateFilesystemFailed", () => Effect.void))
    return yield* recoverMacUpdateTransaction(installed, installedName, staging)
  }))
