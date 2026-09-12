import { lstat } from "node:fs/promises"
import { join } from "node:path"
import { Effect, Option, Schema } from "effect"
import { ExactProcessSchema } from "@magnitudedev/utils/process-groups"
import { SqliteDriver } from "../sqlite-driver"

/** Frozen migration input, independent of the removed runtime election contract. */
export const LegacyOwner = Schema.Struct({
  ...ExactProcessSchema.fields,
  port: Schema.Int.pipe(Schema.between(1, 65535)),
})
export type LegacyOwner = typeof LegacyOwner.Type
const LegacyRow = Schema.Struct({
  id: Schema.Literal(1),
  pid: LegacyOwner.fields.pid,
  process_start_identity: LegacyOwner.fields.processStartIdentity,
  port: LegacyOwner.fields.port,
})

export class LegacyOwnerReadFailed extends Schema.TaggedError<LegacyOwnerReadFailed>()("LegacyOwnerReadFailed", {
  path: Schema.String, message: Schema.String,
}) {}
export class LegacyOwnerInvalid extends Schema.TaggedError<LegacyOwnerInvalid>()("LegacyOwnerInvalid", {
  path: Schema.String, message: Schema.String,
}) {}

/** Reads an existing legacy database; never initializes, claims, repairs, or deletes it. */
export const readLegacyOwner = (dataDirectory: string) => Effect.scoped(Effect.gen(function* () {
  const path = join(dataDirectory, "acn", "coordination.sqlite")
  const info = yield* Effect.tryPromise({
    try: () => lstat(path),
    catch: error => error instanceof Error && "code" in error && error.code === "ENOENT"
      ? new LegacyOwnerMissing() : new LegacyOwnerReadFailed({ path, message: String(error) }),
  }).pipe(Effect.map(Option.some), Effect.catchTag("LegacyOwnerMissing", () => Effect.succeed(Option.none())))
  if (Option.isNone(info)) return Option.none<LegacyOwner>()
  if (!info.value.isFile() || info.value.isSymbolicLink() || (process.platform !== "win32" && info.value.uid !== process.getuid!())) {
    return yield* new LegacyOwnerInvalid({ path, message: "Legacy ownership database must be a regular file owned by this user" })
  }
  const driver = yield* SqliteDriver
  const connection = yield* driver.open(path, { create: false, readOnly: true }).pipe(
    Effect.mapError(error => new LegacyOwnerReadFailed({ path, message: error.message })),
  )
  yield* connection.execute("PRAGMA query_only = ON").pipe(
    Effect.mapError(error => new LegacyOwnerReadFailed({ path, message: error._tag === "SqliteDriverBusy" ? "Legacy database is busy" : error.message })),
  )
  const rows = yield* connection.query("SELECT id, pid, process_start_identity, port FROM owner LIMIT 2").pipe(
    Effect.mapError(error => new LegacyOwnerReadFailed({ path, message: error._tag === "SqliteDriverBusy" ? "Legacy database is busy" : error.message })),
  )
  const owners = yield* Schema.decodeUnknown(Schema.Array(LegacyRow).pipe(Schema.maxItems(1)))(rows).pipe(
    Effect.mapError(() => new LegacyOwnerInvalid({ path, message: "Legacy database contains an invalid owner record" })),
  )
  const row = owners[0]
  return row === undefined ? Option.none<LegacyOwner>() : Option.some(LegacyOwner.make({
    pid: row.pid, processStartIdentity: row.process_start_identity, port: row.port,
  }))
}))

class LegacyOwnerMissing extends Schema.TaggedError<LegacyOwnerMissing>()("LegacyOwnerMissing", {}) {}
