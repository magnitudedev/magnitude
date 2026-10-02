import { Database, SQLiteError, type SQLQueryBindings } from "bun:sqlite"
import { Effect, Layer } from "effect"
import {
  SqliteDriver,
  SqliteDriverBusy,
  type SqliteBinding,
  type SqliteConnection,
  type SqliteDriverError,
  SqliteDriverFailure,
} from "./driver"

const message = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause)

const failure = (cause: unknown): SqliteDriverError =>
  cause instanceof SQLiteError && cause.code === "SQLITE_BUSY"
    ? new SqliteDriverBusy()
    : new SqliteDriverFailure({ message: message(cause) })

const bindings = (values: readonly SqliteBinding[]): SQLQueryBindings[] =>
  Array.from(values) as SQLQueryBindings[]

const connection = (database: Database): SqliteConnection => ({
  execute: (sql, values = []) => Effect.try({
    try: () => {
      database.query(sql).run(...bindings(values))
    },
    catch: failure,
  }),
  query: (sql, values = []) => Effect.try({
    try: () => database.query(sql).all(...bindings(values)),
    catch: failure,
  }),
})

export const BunSqliteDriver: SqliteDriver = {
  open: (path, options) => Effect.acquireRelease(
    Effect.try({
      try: () => new Database(path, options.readOnly ? { readonly: true, create: false } : options.create
        ? { create: true }
        : { create: false, readwrite: true }),
      catch: (cause) => new SqliteDriverFailure({
        message: message(cause),
      }),
    }),
    (database) => Effect.sync(() => database.close()),
  ).pipe(Effect.map(connection)),
}

export const BunSqliteDriverLayer = Layer.succeed(SqliteDriver, BunSqliteDriver)
