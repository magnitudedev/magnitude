import { FileSystem } from "@effect/platform"
import { Context, Effect, Option, Schema, Stream } from "effect"
import { createHash, randomUUID, type KeyObject } from "node:crypto"
import { join, dirname, basename } from "node:path"
import { UpdateOutcome, UpdateRelease, verifyUpdateRelease, updateInstallerFilename, type ReleaseTarget } from "@magnitudedev/release/hosted-update"
import { PrivateFilePermissions } from "./private-files"

export const UpdateInstallation = Schema.Union(
  Schema.TaggedStruct("Unattempted", {}),
  Schema.TaggedStruct("Attempted", {}),
  Schema.TaggedStruct("Failed", { reason: Schema.NonEmptyString.pipe(Schema.maxLength(500)) }),
)
export const PreparedUpdate = Schema.Struct({ release: UpdateRelease, installation: UpdateInstallation })
export type PreparedUpdate = typeof PreparedUpdate.Type
/** Result of the last prepared update. Sent with the next update check and kept so it is reported once. */
export const UpdateOutcomeRecord = Schema.Struct({ outcome: UpdateOutcome, reported: Schema.Boolean })
export type UpdateOutcomeRecord = typeof UpdateOutcomeRecord.Type
export class PreparedUpdateFailed extends Schema.TaggedError<PreparedUpdateFailed>()("PreparedUpdateFailed", {
  message: Schema.String,
}) {}

export interface PreparedUpdateStore {
  readonly read: Effect.Effect<Option.Option<PreparedUpdate>, PreparedUpdateFailed>
  readonly prepare: (archive: string, release: UpdateRelease) => Effect.Effect<void, PreparedUpdateFailed>
  readonly verify: (release: UpdateRelease) => Effect.Effect<string, PreparedUpdateFailed>
  readonly recordAttempt: (release: UpdateRelease) => Effect.Effect<void, PreparedUpdateFailed>
  readonly recordFailure: (release: UpdateRelease, reason: string) => Effect.Effect<void, PreparedUpdateFailed>
  readonly discard: Effect.Effect<void, PreparedUpdateFailed>
  readonly removeAbandonedTransfers: Effect.Effect<void, PreparedUpdateFailed>
  readonly outcome: Effect.Effect<Option.Option<UpdateOutcome>, PreparedUpdateFailed>
  readonly recordOutcome: (outcome: UpdateOutcome) => Effect.Effect<void, PreparedUpdateFailed>
  readonly markOutcomeReported: Effect.Effect<void, PreparedUpdateFailed>
}
export const PreparedUpdateStore = Context.GenericTag<PreparedUpdateStore>("daemon-management/PreparedUpdateStore")

const failed = (message: string) => new PreparedUpdateFailed({ message })
const failureState = (reason: string): PreparedUpdate["installation"] => ({ _tag: "Failed", reason: reason.trim().slice(0, 500) || "The update did not complete." })

const makePreparedUpdateRecord = (dataDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const permissions = yield* PrivateFilePermissions
  const directory = join(dataDirectory, "updates")
  const equalRelease = Schema.equivalence(UpdateRelease)
  const syncDirectory = process.platform === "win32" ? Effect.void
    : Effect.scoped(fs.open(directory, { flag: "r" }).pipe(Effect.flatMap(file => file.sync)))
  const record = <A, I>(name: string, schema: Schema.Schema<A, I>) => {
    const path = join(directory, name)
    const read = Effect.gen(function* () {
      if (!(yield* fs.exists(path))) return Option.none<A>()
      const info = yield* fs.stat(path)
      if (info.type !== "File" || info.size > 4096n || (yield* fs.realPath(path)) !== join(yield* fs.realPath(directory), name)) {
        return yield* failed("The saved update record is invalid.")
      }
      const bytes = yield* fs.stream(path, { bytesToRead: 4097 }).pipe(Stream.runFold(Buffer.alloc(0), (all, chunk) => Buffer.concat([all, chunk])))
      if (bytes.length > 4096) return yield* failed("The saved update record is too large.")
      const text = yield* Effect.try(() => new TextDecoder("utf-8", { fatal: true }).decode(bytes))
      return Option.some(yield* Schema.decodeUnknown(Schema.parseJson(schema))(text, { onExcessProperty: "error" }))
    })
    const write = (value: A) => Effect.gen(function* () {
      const text = yield* Schema.encode(Schema.parseJson(schema))(value)
      yield* permissions.prepareDirectory(directory)
      const temporary = join(directory, `update-${randomUUID()}.tmp`)
      yield* Effect.acquireUseRelease(permissions.createFile(temporary), () => Effect.gen(function* () {
        yield* Effect.scoped(Effect.gen(function* () {
          const file = yield* fs.open(temporary, { flag: "r+" })
          yield* file.writeAll(Buffer.from(text))
          yield* file.sync
        }))
        yield* fs.rename(temporary, path)
        yield* syncDirectory
      }), () => fs.remove(temporary, { force: true }).pipe(Effect.ignore))
    }).pipe(Effect.uninterruptible)
    return { path, read, write }
  }
  const prepared = record("update.json", PreparedUpdate)
  const metadata = prepared.path
  const read = prepared.read.pipe(Effect.mapError(() => failed("The saved update could not be read. Download it again before installing.")))
  const write = (value: PreparedUpdate) => prepared.write(value).pipe(
    Effect.mapError(() => failed("The update state could not be saved. Installation has not been authorized.")))

  const change = (release: UpdateRelease, installation: PreparedUpdate["installation"]) => Effect.gen(function* () {
    const current = yield* read
    if (Option.isNone(current) || !equalRelease(current.value.release, release)) return yield* failed("The prepared update has changed.")
    yield* write({ release, installation })
  }).pipe(Effect.uninterruptible)

  const outcomes = record("outcome.json", UpdateOutcomeRecord)
  const equalOutcome = Schema.equivalence(UpdateOutcome)
  const readOutcome = outcomes.read.pipe(Effect.catchAll(() => fs.remove(outcomes.path, { force: true }).pipe(Effect.ignore, Effect.as(Option.none<UpdateOutcomeRecord>()))))
  const outcome = readOutcome.pipe(Effect.map(Option.flatMap(record => record.reported ? Option.none() : Option.some(record.outcome))))
  const recordOutcome = (value: UpdateOutcome) => Effect.gen(function* () {
    const current = yield* readOutcome
    if (Option.isSome(current) && equalOutcome(current.value.outcome, value)) return
    yield* outcomes.write({ outcome: value, reported: false })
  }).pipe(Effect.mapError(() => failed("The update result could not be saved.")))
  const markOutcomeReported = Effect.gen(function* () {
    const current = yield* readOutcome
    if (Option.isNone(current) || current.value.reported) return
    yield* outcomes.write({ ...current.value, reported: true })
  }).pipe(Effect.mapError(() => failed("The update result could not be saved.")))

  return { read, write, change, directory, metadata, syncDirectory, outcome, recordOutcome, markOutcomeReported }
})

/** The helper records only its exact attempt, while retaining the native installation lease. */
export const recordPreparedUpdateFailure = (dataDirectory: string, release: UpdateRelease, reason: string) =>
  makePreparedUpdateRecord(dataDirectory).pipe(Effect.flatMap(record => record.change(release, failureState(reason))))

/** Call only under application ownership or the native installer handoff lease. No file acts as a lock. */
export const makePreparedUpdateStore = (options: {
  readonly dataDirectory: string
  readonly target: ReleaseTarget
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const permissions = yield* PrivateFilePermissions
  const { read, write, change, directory, metadata, syncDirectory, outcome, recordOutcome, markOutcomeReported } = yield* makePreparedUpdateRecord(options.dataDirectory)
  const installer = join(directory, updateInstallerFilename(options.target))
  const verifyFile = (path: string, release: UpdateRelease) => Effect.gen(function* () {
    yield* verifyUpdateRelease(release, options.target, options.trustedPublishers)
    const info = yield* fs.stat(path)
    if (info.type !== "File" || info.size !== BigInt(release.bytes) || (yield* fs.realPath(path)) !== join(yield* fs.realPath(dirname(path)), basename(path))) {
      return yield* failed("The downloaded update is missing or has changed.")
    }
    const digest = createHash("sha256")
    let length = 0
    yield* fs.stream(path).pipe(Stream.runForEach(chunk => Effect.gen(function* () {
      length += chunk.length
      if (length > release.bytes) return yield* failed("The downloaded update changed during verification.")
      digest.update(chunk)
    })))
    if (length !== release.bytes || digest.digest("hex") !== release.sha256) return yield* failed("The downloaded update failed publisher verification.")
    return path
  }).pipe(Effect.mapError(() => failed("The downloaded update could not be verified. Download it again before installing.")))

  return PreparedUpdateStore.of({
    read, outcome, recordOutcome, markOutcomeReported,
    verify: release => verifyFile(installer, release),
    recordAttempt: release => change(release, { _tag: "Attempted" }),
    recordFailure: (release, reason) => change(release, failureState(reason)),
    prepare: (archive, release) => Effect.gen(function* () {
      yield* verifyUpdateRelease(release, options.target, options.trustedPublishers)
      if (Option.isSome(yield* read)) return yield* failed("An update is already prepared.")
      yield* permissions.prepareDirectory(directory)
      const temporary = join(directory, `installer-${randomUUID()}.tmp`)
      yield* Effect.acquireUseRelease(permissions.createFile(temporary), () => Effect.gen(function* () {
        yield* fs.stream(archive, { bytesToRead: release.bytes + 1 }).pipe(Stream.run(fs.sink(temporary, { flag: "r+" })))
        yield* verifyFile(temporary, release)
        yield* Effect.scoped(fs.open(temporary, { flag: "r+" }).pipe(Effect.flatMap(file => file.sync)))
        yield* Effect.gen(function* () {
          yield* fs.rename(temporary, installer)
          yield* syncDirectory
          yield* write({ release, installation: { _tag: "Unattempted" } })
        }).pipe(Effect.uninterruptible)
      }), () => fs.remove(temporary, { force: true }).pipe(Effect.ignore))
    }).pipe(Effect.mapError(error => error instanceof PreparedUpdateFailed ? error : failed("The downloaded update could not be saved."))),
    discard: Effect.gen(function* () {
      // Metadata is last: an interrupted cleanup remains recognizable on the next launch.
      yield* fs.remove(installer, { force: true })
      yield* fs.remove(metadata, { force: true })
      if (yield* fs.exists(directory)) yield* syncDirectory
    }).pipe(Effect.uninterruptible, Effect.mapError(() => failed("The prepared update could not be removed."))),
    removeAbandonedTransfers: Effect.gen(function* () {
      const transfers = join(options.dataDirectory, "update-downloads")
      if (yield* fs.exists(transfers)) {
        for (const name of yield* fs.readDirectory(transfers)) {
          if (/^desktop-update-[a-zA-Z0-9]+$/.test(name)) yield* fs.remove(join(transfers, name), { recursive: true, force: true })
        }
      }
      if (!(yield* fs.exists(directory))) return
      for (const name of yield* fs.readDirectory(directory)) {
        if (/^(?:update|installer)-[a-f0-9-]{36}\.tmp$/.test(name) || /^desktop-update-[a-zA-Z0-9]+$/.test(name)) {
          yield* fs.remove(join(directory, name), { recursive: true, force: true })
        }
      }
      if (Option.isNone(yield* read)) yield* fs.remove(installer, { force: true })
    }).pipe(Effect.mapError(() => failed("An interrupted update download could not be cleaned up."))),
  })
})
