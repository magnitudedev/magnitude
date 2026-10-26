import { Context, Data, Effect, type Scope } from "effect"

export type SqliteBinding = string | number | bigint | boolean | null | Uint8Array

export class SqliteDriverBusy extends Data.TaggedError("SqliteDriverBusy")<{}> {}

/** The file is not a SQLite database. */
export class SqliteDriverNotADatabase extends Data.TaggedError("SqliteDriverNotADatabase")<{}> {}

export class SqliteDriverFailure extends Data.TaggedError("SqliteDriverFailure")<{
  readonly message: string
}> {}

export type SqliteDriverError = SqliteDriverBusy | SqliteDriverNotADatabase | SqliteDriverFailure

/** A failed SQLite call's error, by its primary result code: the low byte of its extended code. */
export const sqliteError = (code: number, message: string): SqliteDriverError => {
  switch (code & 0xff) {
    case 5: // SQLITE_BUSY
      return new SqliteDriverBusy()
    case 26: // SQLITE_NOTADB
      return new SqliteDriverNotADatabase()
    default:
      return new SqliteDriverFailure({ message })
  }
}

export interface SqliteConnection {
  readonly execute: (
    sql: string,
    bindings?: readonly SqliteBinding[],
  ) => Effect.Effect<void, SqliteDriverError>
  readonly query: (
    sql: string,
    bindings?: readonly SqliteBinding[],
  ) => Effect.Effect<readonly unknown[], SqliteDriverError>
}

export interface SqliteDriver {
  readonly open: (
    path: string,
    options: { readonly create: boolean; readonly readOnly?: false } | { readonly create: false; readonly readOnly: true },
  ) => Effect.Effect<SqliteConnection, SqliteDriverFailure, Scope.Scope>
}

export const SqliteDriver = Context.GenericTag<SqliteDriver>(
  "@magnitudedev/daemon-management/SqliteDriver",
)
