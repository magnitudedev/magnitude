import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Schema } from "effect"
import {
  AcnProcessStoreBusy,
  AcnProcessStoreInvalid,
  AcnProcessStoreUnavailable,
  type AcnProcessStoreError,
} from "./errors"
import { AcnOwnerRecordSchema, type AcnOwnerRecord } from "./schemas"
import {
  SqliteDriver,
  SqliteDriverBusy,
  type SqliteConnection,
  type SqliteDriverError,
} from "./sqlite-driver"

const Sql = {
  busyTimeout: "PRAGMA busy_timeout = 0",
  journalMode: "PRAGMA journal_mode = DELETE",
  createOwner: `CREATE TABLE IF NOT EXISTS owner (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    pid INTEGER NOT NULL CHECK (pid > 0 AND pid <= 9007199254740991),
    process_start_identity TEXT NOT NULL
      CHECK (length(process_start_identity) > 0),
    port INTEGER NOT NULL CHECK (port BETWEEN 1 AND 65535)
  )`,
  currentOwner: `SELECT pid, process_start_identity, port
    FROM owner WHERE id = 1`,
  ownerCount: "SELECT COUNT(*) AS count FROM owner",
  replaceOwner: `INSERT INTO owner (id, pid, process_start_identity, port)
    VALUES (1, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      pid = excluded.pid,
      process_start_identity = excluded.process_start_identity,
      port = excluded.port`,
  beginImmediate: "BEGIN IMMEDIATE",
  commit: "COMMIT",
  rollback: "ROLLBACK",
} as const

const OwnerRowSchema = Schema.Struct({
  pid: AcnOwnerRecordSchema.fields.pid,
  process_start_identity: AcnOwnerRecordSchema.fields.processStartIdentity,
  port: AcnOwnerRecordSchema.fields.port,
})

const CountRowSchema = Schema.Struct({ count: Schema.Number.pipe(Schema.int(), Schema.nonNegative()) })

const invalid = (path: string, message: string) =>
  new AcnProcessStoreInvalid({ path, message })

const storeError = (
  operation: string,
  path: string,
  error: SqliteDriverError,
): AcnProcessStoreError => error instanceof SqliteDriverBusy
  ? new AcnProcessStoreBusy({ operation, path })
  : new AcnProcessStoreUnavailable({ operation, path, message: error.message })

const decodeOne = <A, I>(
  schema: Schema.Schema<A, I>,
  path: string,
  description: string,
  rows: readonly unknown[],
): Effect.Effect<A, AcnProcessStoreInvalid> => rows.length === 1
  ? Schema.decodeUnknown(schema)(rows[0]).pipe(
      Effect.mapError((error) => invalid(path, `${description} is malformed: ${String(error)}`)),
    )
  : Effect.fail(invalid(path, `${description} returned ${rows.length} rows`))

const ownerFromRow = (row: typeof OwnerRowSchema.Type): AcnOwnerRecord =>
  AcnOwnerRecordSchema.make({
    pid: row.pid,
    processStartIdentity: row.process_start_identity,
    port: row.port,
  })

const sameOwner = (
  left: Option.Option<AcnOwnerRecord>,
  right: Option.Option<AcnOwnerRecord>,
): boolean => Option.match(left, {
  onNone: () => Option.isNone(right),
  onSome: (owner) => Option.exists(right, (other) =>
    owner.pid === other.pid &&
    owner.processStartIdentity === other.processStartIdentity &&
    owner.port === other.port),
})

export type ReplaceOwnerResult =
  | { readonly _tag: "Replaced" }
  | { readonly _tag: "OwnerChanged"; readonly owner: Option.Option<AcnOwnerRecord> }

export interface AcnCoordinationDatabase {
  readonly currentOwner: Effect.Effect<Option.Option<AcnOwnerRecord>, AcnProcessStoreError>
  readonly replaceOwner: (
    expectedOwner: Option.Option<AcnOwnerRecord>,
    candidateOwner: AcnOwnerRecord,
  ) => Effect.Effect<ReplaceOwnerResult, AcnProcessStoreError>
}

export const makeAcnCoordinationDatabase = (
  dataDirectory: string,
): Effect.Effect<
  AcnCoordinationDatabase,
  never,
  FileSystem.FileSystem | Path.Path | SqliteDriver
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const paths = yield* Path.Path
  const driver = yield* SqliteDriver
  const directory = paths.join(dataDirectory, "acn")
  const databasePath = paths.join(directory, "coordination.sqlite")

  const withConnection = <A>(
    operation: string,
    use: (connection: SqliteConnection) => Effect.Effect<A, AcnProcessStoreError>,
  ): Effect.Effect<A, AcnProcessStoreError> => Effect.scoped(Effect.gen(function* () {
    yield* fs.makeDirectory(directory, { recursive: true }).pipe(
      Effect.mapError((error) => new AcnProcessStoreUnavailable({
        operation: "create-directory",
        path: directory,
        message: String(error),
      })),
    )
    const connection = yield* driver.open(databasePath, { create: true }).pipe(
      Effect.mapError((error) => storeError(operation, databasePath, error)),
    )
    const execute = (sql: string) => connection.execute(sql).pipe(
      Effect.mapError((error) => storeError(operation, databasePath, error)),
    )
    yield* execute(Sql.busyTimeout)
    yield* execute(Sql.journalMode)
    yield* execute(Sql.createOwner)
    return yield* use(connection)
  }))

  const queryOwner = (
    connection: SqliteConnection,
    operation: string,
  ): Effect.Effect<Option.Option<AcnOwnerRecord>, AcnProcessStoreError> =>
    Effect.gen(function* () {
      const count = yield* connection.query(Sql.ownerCount).pipe(
        Effect.mapError((error) => storeError(operation, databasePath, error)),
        Effect.flatMap((rows) => decodeOne(CountRowSchema, databasePath, "owner count query", rows)),
      )
      if (count.count > 1) return yield* invalid(databasePath, "owner contains multiple rows")
      if (count.count === 0) return Option.none()
      const rows = yield* connection.query(Sql.currentOwner).pipe(
        Effect.mapError((error) => storeError(operation, databasePath, error)),
      )
      return Option.some(ownerFromRow(yield* decodeOne(
        OwnerRowSchema,
        databasePath,
        "owner query",
        rows,
      )))
    })

  return {
    currentOwner: withConnection("current-owner", (connection) =>
      queryOwner(connection, "current-owner")),
    replaceOwner: (expectedOwner, candidateOwner) =>
      withConnection("replace-owner", (connection) => Effect.uninterruptibleMask(() =>
        Effect.gen(function* () {
          yield* connection.execute(Sql.beginImmediate).pipe(
            Effect.mapError((error) => storeError("replace-owner", databasePath, error)),
          )
          const transaction = Effect.gen(function* () {
            const owner = yield* queryOwner(connection, "replace-owner")
            if (!sameOwner(owner, expectedOwner)) {
              yield* connection.execute(Sql.rollback).pipe(
                Effect.mapError((error) => storeError("replace-owner", databasePath, error)),
              )
              return { _tag: "OwnerChanged" as const, owner }
            }
            yield* connection.execute(Sql.replaceOwner, [
              candidateOwner.pid,
              candidateOwner.processStartIdentity,
              candidateOwner.port,
            ]).pipe(Effect.mapError((error) => storeError("replace-owner", databasePath, error)))
            yield* connection.execute(Sql.commit).pipe(
              Effect.mapError((error) => storeError("replace-owner", databasePath, error)),
            )
            return { _tag: "Replaced" as const }
          })
          return yield* transaction.pipe(
            Effect.onError(() => connection.execute(Sql.rollback).pipe(Effect.ignore)),
          )
        }),
      )),
  }
})
