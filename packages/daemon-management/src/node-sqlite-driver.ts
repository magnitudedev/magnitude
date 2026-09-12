import { DatabaseSync, type SQLInputValue } from "node:sqlite"
import { pathToFileURL } from "node:url"
import { Effect, Layer } from "effect"
import { SqliteDriver, SqliteDriverBusy, SqliteDriverFailure, type SqliteBinding, type SqliteConnection, type SqliteDriverError } from "./sqlite-driver"

const failure = (cause: unknown): SqliteDriverError => {
  // SQLite extended result codes retain the primary code in the low byte.
  if (cause instanceof Error && "errcode" in cause && typeof cause.errcode === "number" && (cause.errcode & 0xff) === 5) {
    return new SqliteDriverBusy()
  }
  return new SqliteDriverFailure({ message: cause instanceof Error ? cause.message : String(cause) })
}
const bindings = (values: readonly SqliteBinding[]): SQLInputValue[] => values.map((value) => typeof value === "boolean" ? Number(value) : value)
const connection = (database: DatabaseSync): SqliteConnection => ({
  execute: (sql, values = []) => Effect.try({
    try: () => { database.prepare(sql).run(...bindings(values)) },
    catch: failure,
  }),
  query: (sql, values = []) => Effect.try({
    try: () => database.prepare(sql).all(...bindings(values)),
    catch: failure,
  }),
})

/** Electron's Node runtime owns this adapter; Bun hosts use their own adapter. */
export const NodeSqliteDriver: SqliteDriver = {
  open: (path, options) => Effect.acquireRelease(
    Effect.try({
      // URI mode=rw makes SQLite itself refuse missing files, without a check/open race.
      try: () => new DatabaseSync(`${pathToFileURL(path).href}?mode=${options.readOnly ? "ro" : options.create ? "rwc" : "rw"}`, { timeout: 0, readOnly: options.readOnly ?? false }),
      catch: (cause) => new SqliteDriverFailure({ message: cause instanceof Error ? cause.message : String(cause) }),
    }),
    (database) => Effect.sync(() => database.close()),
  ).pipe(Effect.map(connection)),
}
export const NodeSqliteDriverLayer = Layer.succeed(SqliteDriver, NodeSqliteDriver)
