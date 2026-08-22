import { Context, Data, Effect, type Scope } from "effect"

export type SqliteBinding = string | number | bigint | boolean | null | Uint8Array

export class SqliteDriverBusy extends Data.TaggedError("SqliteDriverBusy")<{}> {}

export class SqliteDriverFailure extends Data.TaggedError("SqliteDriverFailure")<{
  readonly message: string
}> {}

export type SqliteDriverError = SqliteDriverBusy | SqliteDriverFailure

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
    options: { readonly create: boolean },
  ) => Effect.Effect<SqliteConnection, SqliteDriverFailure, Scope.Scope>
}

export const SqliteDriver = Context.GenericTag<SqliteDriver>(
  "@magnitudedev/acn-protocol/coordination/SqliteDriver",
)
