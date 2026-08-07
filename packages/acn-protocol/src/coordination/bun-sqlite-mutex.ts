import { Database, SQLiteError } from "bun:sqlite"
import { Effect } from "effect"
import {
  makeSqliteMutex,
  SqliteMutexBusy,
  type SqliteMutexConnection,
  type SqliteMutexDriver,
  SqliteMutexDriverError,
  type SqliteMutexDriverFailure,
} from "./sqlite-mutex"

const failure = (error: Error): SqliteMutexDriverFailure =>
  error instanceof SQLiteError && error.code === "SQLITE_BUSY"
    ? new SqliteMutexBusy()
    : new SqliteMutexDriverError({ message: error.message })

const tryDriver = <A>(evaluate: () => A): Effect.Effect<A, SqliteMutexDriverFailure> =>
  Effect.try({
    try: evaluate,
    catch: (cause) => failure(cause instanceof Error ? cause : new Error(String(cause))),
  })

const connection = (database: Database): SqliteMutexConnection => ({
  run: (sql) => tryDriver(() => {
    database.run(sql)
  }),
  get: <Row>(sql: string) => tryDriver(() => database.query<Row, []>(sql).get() ?? undefined),
  close: Effect.try({
    try: () => database.close(),
    catch: (cause) => new SqliteMutexDriverError({
      message: cause instanceof Error ? cause.message : String(cause),
    }),
  }),
})

const BunSqliteMutexDriver: SqliteMutexDriver = {
  open: (path, options) => Effect.try({
    try: () => connection(new Database(path, options.create
      ? { create: true }
      : { create: false, readwrite: true })),
    catch: (cause) => new SqliteMutexDriverError({
      message: cause instanceof Error ? cause.message : String(cause),
    }),
  }),
}

export const BunSqliteMutex = makeSqliteMutex(BunSqliteMutexDriver)
