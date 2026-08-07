import { Context, Data, Effect, Scope } from "effect"

const Sql = {
  setInitializationBusyTimeout: "PRAGMA busy_timeout = 5000",
  setAcquisitionBusyTimeout: "PRAGMA busy_timeout = 0",
  createAnchor: `CREATE TABLE IF NOT EXISTS magnitude_mutex (
    id INTEGER PRIMARY KEY CHECK (id = 1)
  )`,
  insertAnchor: "INSERT OR IGNORE INTO magnitude_mutex (id) VALUES (1)",
  selectAnchorTable: `SELECT 1 AS present FROM sqlite_master
    WHERE type = 'table' AND name = 'magnitude_mutex'`,
  selectAnchor: "SELECT id FROM magnitude_mutex WHERE id = 1",
  begin: "BEGIN IMMEDIATE",
  rollback: "ROLLBACK",
} as const

export type SqliteMutexOperation = "initialize" | "acquire" | "release"

export class SqliteMutexError extends Data.TaggedError("SqliteMutexError")<{
  readonly operation: SqliteMutexOperation
  readonly path: string
  readonly message: string
}> {}

export class SqliteMutexDriverError extends Data.TaggedError("SqliteMutexDriverError")<{
  readonly message: string
}> {}

export class SqliteMutexBusy extends Data.TaggedError("SqliteMutexBusy") {}

export type SqliteMutexDriverFailure = SqliteMutexDriverError | SqliteMutexBusy

export interface SqliteMutexConnection {
  readonly run: (sql: string) => Effect.Effect<void, SqliteMutexDriverFailure>
  readonly get: <Row>(sql: string) => Effect.Effect<Row | undefined, SqliteMutexDriverFailure>
  readonly close: Effect.Effect<void, SqliteMutexDriverError>
}

export interface SqliteMutexDriver {
  readonly open: (
    path: string,
    options: { readonly create: boolean },
  ) => Effect.Effect<SqliteMutexConnection, SqliteMutexDriverError>
}

export interface SqliteMutex {
  readonly initialize: (path: string) => Effect.Effect<void, SqliteMutexError>
  readonly tryAcquire: (
    path: string,
  ) => Effect.Effect<boolean, SqliteMutexError, Scope.Scope>
}

export const SqliteMutex = Context.GenericTag<SqliteMutex>(
  "@magnitudedev/acn-protocol/coordination/SqliteMutex",
)

const publicError = (
  operation: SqliteMutexOperation,
  path: string,
  error: SqliteMutexDriverError | SqliteMutexBusy,
): SqliteMutexError => new SqliteMutexError({
  operation,
  path,
  message: error._tag === "SqliteMutexBusy"
    ? "SQLite mutex remained busy"
    : error.message,
})

const anchorExists = (connection: SqliteMutexConnection) => connection
  .get<{ readonly id: number }>(Sql.selectAnchor)
  .pipe(Effect.map((row) => row?.id === 1))

const anchorTableExists = (connection: SqliteMutexConnection) => connection
  .get<{ readonly present: number }>(Sql.selectAnchorTable)
  .pipe(Effect.map((row) => row?.present === 1))

export const makeSqliteMutex = (driver: SqliteMutexDriver): SqliteMutex => SqliteMutex.of({
  initialize: (path) => Effect.acquireUseRelease(
    driver.open(path, { create: true }),
    (connection) => Effect.gen(function* () {
      yield* connection.run(Sql.setInitializationBusyTimeout)
      if (!(yield* anchorTableExists(connection))) {
        yield* connection.run(Sql.createAnchor)
      }
      if (!(yield* anchorExists(connection))) {
        yield* connection.run(Sql.insertAnchor)
      }
    }),
    (connection) => connection.close.pipe(Effect.orDie),
  ).pipe(Effect.mapError((error) => publicError("initialize", path, error))),

  tryAcquire: (path) => Effect.acquireRelease(
    Effect.gen(function* () {
      const connection = yield* driver.open(path, { create: false })
      const acquired = yield* Effect.gen(function* () {
        yield* connection.run(Sql.setAcquisitionBusyTimeout)
        if (!(yield* anchorExists(connection))) {
          return yield* new SqliteMutexDriverError({
            message: "SQLite mutex anchor is absent",
          })
        }
        yield* connection.run(Sql.begin)
      }).pipe(
        Effect.onError(() => connection.close.pipe(Effect.orDie)),
        Effect.as(true),
        Effect.catchTag("SqliteMutexBusy", () => Effect.succeed(false)),
        Effect.mapError((error) => publicError("acquire", path, error)),
      )
      return acquired ? connection : null
    }).pipe(Effect.mapError((error) => error instanceof SqliteMutexError
      ? error
      : publicError("acquire", path, error))),
    (connection) => connection === null
      ? Effect.void
      : connection.run(Sql.rollback).pipe(
          Effect.ensuring(connection.close.pipe(Effect.orDie)),
          Effect.mapError((error) => publicError("release", path, error)),
          Effect.orDie,
        ),
  ).pipe(Effect.map((connection) => connection !== null)),
})
