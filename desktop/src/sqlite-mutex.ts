import sqlite3 from "sqlite3"
import { Effect, Layer } from "effect"
import {
  makeSqliteMutex,
  SqliteMutex,
  SqliteMutexBusy,
  type SqliteMutexConnection,
  type SqliteMutexDriver,
  SqliteMutexDriverError,
  type SqliteMutexDriverFailure,
} from "@magnitudedev/sdk/sqlite-mutex"

interface Sqlite3Error extends Error {
  readonly code: string
}

const isSqlite3Error = (error: Error): error is Sqlite3Error =>
  "code" in error && typeof error.code === "string"

const failure = (error: Error): SqliteMutexDriverFailure =>
  isSqlite3Error(error) && error.code === "SQLITE_BUSY"
    ? new SqliteMutexBusy()
    : new SqliteMutexDriverError({ message: error.message })

const run = (
  database: sqlite3.Database,
  sql: string,
): Effect.Effect<void, SqliteMutexDriverFailure> => Effect.async((resume) => {
  database.run(sql, (error) => resume(error === null
    ? Effect.void
    : Effect.fail(failure(error))))
})

const get = <Row>(
  database: sqlite3.Database,
  sql: string,
): Effect.Effect<Row | undefined, SqliteMutexDriverFailure> => Effect.async((resume) => {
  database.get<Row>(sql, (error, row) => resume(error === null
    ? Effect.succeed(row)
    : Effect.fail(failure(error))))
})

const close = (
  database: sqlite3.Database,
): Effect.Effect<void, SqliteMutexDriverError> => Effect.async((resume) => {
  database.close((error) => resume(error === null
    ? Effect.void
    : Effect.fail(new SqliteMutexDriverError({ message: error.message }))))
})

const connection = (database: sqlite3.Database): SqliteMutexConnection => ({
  run: (sql) => run(database, sql),
  get: <Row>(sql: string) => get<Row>(database, sql),
  close: close(database),
})

const NodeSqliteMutexDriver: SqliteMutexDriver = {
  open: (path, options) => Effect.async((resume) => {
    const mode = sqlite3.OPEN_READWRITE | (options.create ? sqlite3.OPEN_CREATE : 0)
    const database = new sqlite3.Database(path, mode, (error) => resume(error === null
      ? Effect.succeed(connection(database))
      : Effect.fail(new SqliteMutexDriverError({ message: error.message }))))
  }),
}

export const NodeSqliteMutex = makeSqliteMutex(NodeSqliteMutexDriver)

export const NodeSqliteMutexLayer = Layer.succeed(SqliteMutex, NodeSqliteMutex)
