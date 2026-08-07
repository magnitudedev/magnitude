import sqlite3 from "sqlite3"
import { Effect, Layer } from "effect"
import {
  SqliteDriver,
  SqliteDriverBusy,
  type SqliteBinding,
  type SqliteConnection,
  type SqliteDriverError,
  type SqliteDriverService,
  SqliteDriverFailure,
} from "@magnitudedev/sdk/sqlite-driver"

interface Sqlite3Error extends Error {
  readonly code: string
}

const isSqlite3Error = (error: Error): error is Sqlite3Error =>
  "code" in error && typeof error.code === "string"

const failure = (operation: string, error: Error): SqliteDriverError =>
  isSqlite3Error(error) && error.code === "SQLITE_BUSY"
    ? new SqliteDriverBusy({ operation })
    : new SqliteDriverFailure({ operation, message: error.message })

const execute = (
  database: sqlite3.Database,
  sql: string,
  bindings: readonly SqliteBinding[],
): Effect.Effect<void, SqliteDriverError> => Effect.async((resume) => {
  database.run(sql, [...bindings], (error) => resume(error === null
    ? Effect.void
    : Effect.fail(failure("execute", error))))
  return Effect.sync(() => database.interrupt())
})

const query = (
  database: sqlite3.Database,
  sql: string,
  bindings: readonly SqliteBinding[],
): Effect.Effect<readonly unknown[], SqliteDriverError> => Effect.async((resume) => {
  database.all(sql, [...bindings], (error, rows) => resume(error === null
    ? Effect.succeed(rows)
    : Effect.fail(failure("query", error))))
  return Effect.sync(() => database.interrupt())
})

const close = (database: sqlite3.Database): Effect.Effect<void> =>
  Effect.async((resume) => {
    database.close((error) => resume(error === null ? Effect.void : Effect.die(error)))
  })

const connection = (database: sqlite3.Database): SqliteConnection => ({
  execute: (sql, bindings = []) => execute(database, sql, bindings),
  query: (sql, bindings = []) => query(database, sql, bindings),
})

export const NodeSqliteDriver: SqliteDriverService = {
  open: (path, options) => Effect.acquireRelease(
    Effect.async<sqlite3.Database, SqliteDriverFailure>((resume) => {
      const mode = sqlite3.OPEN_READWRITE | (options.create ? sqlite3.OPEN_CREATE : 0)
      const database = new sqlite3.Database(path, mode, (error) => resume(error === null
        ? Effect.succeed(database)
        : Effect.fail(new SqliteDriverFailure({
            operation: "open",
            message: error.message,
          }))))
      return Effect.sync(() => database.close())
    }),
    close,
  ).pipe(Effect.map(connection)),
}

export const NodeSqliteDriverLayer = Layer.succeed(SqliteDriver, NodeSqliteDriver)
