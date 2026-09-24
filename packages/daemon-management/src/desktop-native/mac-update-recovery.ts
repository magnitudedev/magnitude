import { FSM } from "@magnitudedev/utils"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { MacFileIdentity, MacUpdateFilesystem, type MacUpdateDirectory } from "./mac-update-filesystem"
import { MacBundleVerifier } from "./mac-update-validation"

const BundleName = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(255),
  Schema.filter(name => name !== "." && name !== ".." && !/[\\/:\0]/.test(name)))
const Bundle = Schema.Struct({ identity: MacFileIdentity, version: Schema.NonEmptyString.pipe(Schema.maxLength(96)) })
const Transaction = Schema.Struct({
  id: Schema.UUID.pipe(Schema.brand("MacUpdateTransactionId")),
  installedParent: MacFileIdentity,
  stagingParent: MacFileIdentity,
  installedName: BundleName,
  architecture: Schema.Literal("arm64", "x64"),
  previous: Bundle,
  replacement: Bundle,
}).pipe(Schema.filter(value => value.previous.identity !== value.replacement.identity && value.installedParent !== value.stagingParent))
const fields = { protocol: Schema.Literal(1), transaction: Transaction }
export class ExchangeIntent extends Schema.TaggedClass<ExchangeIntent>()("ExchangeIntent", fields) {}
class Committed extends Schema.TaggedClass<Committed>()("Committed", fields) {}
class RestoreIntent extends Schema.TaggedClass<RestoreIntent>()("RestoreIntent", fields) {}
class Restored extends Schema.TaggedClass<Restored>()("Restored", fields) {}
class Abandoned extends Schema.TaggedClass<Abandoned>()("Abandoned", fields) {}
export const MacUpdateJournal = Schema.Union(ExchangeIntent, Committed, RestoreIntent, Restored, Abandoned)
export type MacUpdateJournal = typeof MacUpdateJournal.Type
const lifecycle = FSM.defineFSM({ ExchangeIntent, Committed, RestoreIntent, Restored, Abandoned }, {
  ExchangeIntent: ["Committed", "RestoreIntent", "Abandoned"],
  RestoreIntent: ["Restored"], Committed: [], Restored: [], Abandoned: [],
} as const)

export const MacUpdateRecoveryResult = Schema.Union(
  Schema.TaggedStruct("NoTransaction", {}),
  Schema.TaggedStruct("Installed", { version: Schema.String }),
  Schema.TaggedStruct("Preserved", { version: Schema.String, reason: Schema.Literal("Interrupted", "Restored") }),
)
export type MacUpdateRecoveryResult = typeof MacUpdateRecoveryResult.Type
export class MacUpdateRepairRequired extends Schema.TaggedError<MacUpdateRepairRequired>()("MacUpdateRepairRequired", {}) {
  override get message() { return "The application update requires repair before Magnitude can start." }
}
const repair = () => new MacUpdateRepairRequired()

const readJournal = (installed: MacUpdateDirectory, installedName: string, staging: MacUpdateDirectory) => Effect.gen(function* () {
  const fs = yield* MacUpdateFilesystem
  const bytes = yield* fs.readRecord(staging)
  if (Option.isNone(bytes)) return Option.none<MacUpdateJournal>()
  const text = yield* Effect.try({ try: () => new TextDecoder("utf-8", { fatal: true }).decode(bytes.value), catch: repair })
  const record = yield* Schema.decodeUnknown(Schema.parseJson(MacUpdateJournal))(text, { onExcessProperty: "error" }).pipe(Effect.mapError(repair))
  if (record.transaction.installedParent !== installed.identity || record.transaction.stagingParent !== staging.identity ||
      record.transaction.installedName !== installedName) return yield* repair()
  return Option.some(record)
}).pipe(Effect.catchTag("MacUpdateFilesystemFailed", repair))

/** Caller holds installation exclusion. Recovery never starts a service or retries a forward exchange. */
export const recoverMacUpdateTransaction = (installed: MacUpdateDirectory, installedName: string, staging: MacUpdateDirectory):
  Effect.Effect<MacUpdateRecoveryResult, MacUpdateRepairRequired, MacUpdateFilesystem | MacBundleVerifier> =>
  Effect.gen(function* () {
    const fs = yield* MacUpdateFilesystem
    const verifier = yield* MacBundleVerifier
    const saved = yield* readJournal(installed, installedName, staging)
    if (Option.isNone(saved)) return { _tag: "NoTransaction" } as const
    const record = saved.value
    const transaction = record.transaction
    const observe = Effect.gen(function* () {
      const current = yield* fs.inspect(installed, installedName)
      const displaced = yield* fs.inspect(staging, "Magnitude.app")
      const currentIs = (identity: MacFileIdentity) => Option.contains(current, identity)
      const stagedIs = (identity: MacFileIdentity) => Option.contains(displaced, identity)
      if (currentIs(transaction.previous.identity) && stagedIs(transaction.replacement.identity)) return "OldAndNew" as const
      if (currentIs(transaction.replacement.identity) && stagedIs(transaction.previous.identity)) return "NewAndOld" as const
      if (currentIs(transaction.previous.identity) && Option.isNone(displaced)) return "OldOnly" as const
      if (currentIs(transaction.replacement.identity) && Option.isNone(displaced)) return "NewOnly" as const
      return yield* repair()
    })
    const verify = (directory: MacUpdateDirectory, name: string, bundle: typeof Bundle.Type) => Effect.gen(function* () {
      if (!Option.contains(yield* fs.inspect(directory, name), bundle.identity)) return yield* repair()
      yield* verifier.verify(join(directory.path, name), { version: bundle.version, architecture: transaction.architecture })
      if (!Option.contains(yield* fs.inspect(directory, name), bundle.identity)) return yield* repair()
    })
    const save = (next: MacUpdateJournal) => Effect.gen(function* () {
      const encoded = yield* Schema.encode(Schema.parseJson(MacUpdateJournal))(next).pipe(Effect.mapError(repair))
      // Reconcile namespace durability even when the exchange's original sync reported failure.
      yield* fs.sync(installed)
      yield* fs.sync(staging)
      yield* fs.writeRecord(staging, Buffer.from(encoded))
    })
    const installedResult = { _tag: "Installed", version: transaction.replacement.version } as const
    const preserved = (reason: "Interrupted" | "Restored") => ({ _tag: "Preserved", version: transaction.previous.version, reason } as const)
    const restore = (intent: RestoreIntent) => Effect.gen(function* () {
      const layout = yield* observe
      if (layout === "NewAndOld") {
        yield* verify(staging, "Magnitude.app", transaction.previous)
        // A failed exchange may already have restored the old bundle. Observe before deciding.
        yield* fs.exchange(installed, installedName, transaction.replacement.identity,
          staging, "Magnitude.app", transaction.previous.identity).pipe(Effect.catchTag("MacUpdateFilesystemFailed", () => Effect.void))
        if ((yield* observe) !== "OldAndNew") return yield* repair()
      } else if (layout !== "OldAndNew") return yield* repair()
      yield* verify(installed, installedName, transaction.previous)
      yield* save(lifecycle.transition(intent, "Restored", {}))
      return preserved("Restored")
    })
    const layout = yield* observe
    switch (record._tag) {
      case "Committed":
        if (layout !== "NewAndOld" && layout !== "NewOnly") return yield* repair()
        yield* verify(installed, installedName, transaction.replacement)
        return installedResult
      case "Abandoned":
      case "Restored":
        if (layout !== "OldAndNew" && layout !== "OldOnly") return yield* repair()
        yield* verify(installed, installedName, transaction.previous)
        return preserved(record._tag === "Restored" ? "Restored" : "Interrupted")
      case "RestoreIntent": return yield* restore(record)
      case "ExchangeIntent": {
        if (layout === "OldAndNew") {
          yield* verify(installed, installedName, transaction.previous)
          yield* save(lifecycle.transition(record, "Abandoned", {}))
          return preserved("Interrupted")
        }
        if (layout !== "NewAndOld") return yield* repair()
        const valid = yield* verify(installed, installedName, transaction.replacement).pipe(
          Effect.as(true), Effect.catchTag("MacBundleVerificationFailed", () => Effect.succeed(false)))
        if (valid) {
          yield* save(lifecycle.transition(record, "Committed", {}))
          return installedResult
        }
        yield* verify(staging, "Magnitude.app", transaction.previous)
        const intent = lifecycle.transition(record, "RestoreIntent", {})
        yield* save(intent)
        return yield* restore(intent)
      }
    }
  }).pipe(
    Effect.catchTags({ MacUpdateFilesystemFailed: () => repair(), MacBundleVerificationFailed: () => repair() }),
    Effect.uninterruptible,
  )

export class MacUpdateCleanupFailed extends Schema.TaggedError<MacUpdateCleanupFailed>()("MacUpdateCleanupFailed", {}) {
  override get message() { return "The completed update has retained files to clean up on the next startup." }
}

/** Retire displaced contents after terminal reconciliation, then remove the exact terminal receipt. */
export const retireMacUpdateBundle = (installed: MacUpdateDirectory, installedName: string, staging: MacUpdateDirectory):
  Effect.Effect<MacUpdateRecoveryResult, MacUpdateRepairRequired | MacUpdateCleanupFailed, MacUpdateFilesystem | MacBundleVerifier> =>
  Effect.gen(function* () {
    const fs = yield* MacUpdateFilesystem
    const saved = yield* readJournal(installed, installedName, staging)
    if (Option.isNone(saved)) {
      yield* fs.sync(staging).pipe(Effect.mapError(() => new MacUpdateCleanupFailed()))
      return { _tag: "NoTransaction" } as const
    }
    const record = saved.value
    if (record._tag === "ExchangeIntent" || record._tag === "RestoreIntent") return yield* repair()
    const result = yield* recoverMacUpdateTransaction(installed, installedName, staging)
    const expected = record._tag === "Committed" ? record.transaction.previous.identity : record.transaction.replacement.identity
    const displaced = yield* fs.inspect(staging, "Magnitude.app").pipe(Effect.mapError(repair))
    if (Option.isSome(displaced)) {
      if (displaced.value !== expected) return yield* repair()
      yield* fs.removeTree(staging, "Magnitude.app", expected).pipe(Effect.mapError(() => new MacUpdateCleanupFailed()))
    }
    // The receipt is last: partial tree deletion remains recoverable against the verified installed bundle.
    yield* fs.sync(staging).pipe(Effect.mapError(() => new MacUpdateCleanupFailed()))
    const receipt = yield* fs.readRecord(staging).pipe(Effect.mapError(() => new MacUpdateCleanupFailed()))
    if (Option.isNone(receipt)) return yield* repair()
    yield* fs.removeRecord(staging, receipt.value).pipe(Effect.mapError(() => new MacUpdateCleanupFailed()))
    return result
  }).pipe(Effect.uninterruptible)
