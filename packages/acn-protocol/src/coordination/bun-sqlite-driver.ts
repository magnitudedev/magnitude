import { Database, SQLiteError, type SQLQueryBindings } from "bun:sqlite"
import { Effect } from "effect"
import {
  SqliteDriverBusy,
  type SqliteBinding,
  type SqliteConnection,
  type SqliteDriver,
  type SqliteDriverError,
  SqliteDriverFailure,
} from "./sqlite-driver"

const message = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause)

const failure = (operation: string, cause: unknown): SqliteDriverError =>
  cause instanceof SQLiteError && cause.code === "SQLITE_BUSY"
    ? new SqliteDriverBusy({ operation })
    : new SqliteDriverFailure({ operation, message: message(cause) })

const bindings = (values: readonly SqliteBinding[]): SQLQueryBindings[] =>
  Array.from(values) as SQLQueryBindings[]

const connection = (database: Database): SqliteConnection => ({
  execute: (sql, values = []) => Effect.try({
    try: () => {
      database.query(sql).run(...bindings(values))
    },
    catch: (cause) => failure("execute", cause),
  }),
  query: (sql, values = []) => Effect.try({
    try: () => database.query(sql).all(...bindings(values)),
    catch: (cause) => failure("query", cause),
  }),
})

export const BunSqliteDriver: SqliteDriver = {
  open: (path, options) => Effect.acquireRelease(
    Effect.try({
      try: () => new Database(path, options.create
        ? { create: true }
        : { create: false, readwrite: true }),
      catch: (cause) => new SqliteDriverFailure({
        operation: "open",
        message: message(cause),
      }),
    }),
    (database) => Effect.sync(() => database.close()),
  ).pipe(Effect.map(connection)),
}
