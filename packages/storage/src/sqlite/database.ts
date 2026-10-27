import { Cause, Data, Effect, Exit, type ParseResult, Schema } from "effect"
import type { SqliteConnection, SqliteDriverError } from "./driver"

const UserVersion = Schema.Tuple(Schema.Struct({ user_version: Schema.Number }))
const JournalMode = Schema.Tuple(Schema.Struct({ journal_mode: Schema.String }))
const Checkpoint = Schema.Tuple(Schema.Struct({ busy: Schema.Number, log: Schema.Number, checkpointed: Schema.Number }))

/** SQLite kept another journal mode when asked for a write-ahead log. */
export class SqliteWalRefused extends Data.TaggedError("SqliteWalRefused")<{
  readonly journalMode: string
}> {}

/** A checkpoint left part of the write-ahead log: `busy` is 1 when a connection blocked it, and `checkpointed` of its `log` frames were copied. */
export class SqliteCheckpointIncomplete extends Data.TaggedError("SqliteCheckpointIncomplete")<{
  readonly busy: number
  readonly log: number
  readonly checkpointed: number
}> {}

/** The file's schema version; 0 for a new file. */
export const userVersion = (db: SqliteConnection): Effect.Effect<number, SqliteDriverError | ParseResult.ParseError> =>
  db.query("PRAGMA user_version").pipe(
    Effect.flatMap(Schema.decodeUnknown(UserVersion)),
    Effect.map(([row]) => row.user_version)
  )

/** Puts the file in write-ahead-log mode. SQLite answers with the mode it is in, which may be another. */
export const enableWal = (db: SqliteConnection): Effect.Effect<void, SqliteDriverError | ParseResult.ParseError | SqliteWalRefused> =>
  db.query("PRAGMA journal_mode=WAL").pipe(
    Effect.flatMap(Schema.decodeUnknown(JournalMode)),
    Effect.flatMap(([{ journal_mode }]) => journal_mode === "wal" ? Effect.void : new SqliteWalRefused({ journalMode: journal_mode }))
  )

/** Copies the whole write-ahead log into the database and truncates it, failing unless every frame was copied. */
export const truncateWal = (db: SqliteConnection): Effect.Effect<void, SqliteDriverError | ParseResult.ParseError | SqliteCheckpointIncomplete> =>
  db.query("PRAGMA wal_checkpoint(TRUNCATE)").pipe(
    Effect.flatMap(Schema.decodeUnknown(Checkpoint)),
    Effect.flatMap(([row]) => row.busy === 0 && row.checkpointed === row.log ? Effect.void : new SqliteCheckpointIncomplete(row))
  )

// Ends a failed transaction, keeping its failure and any failure to roll it back.
const rollback = <E>(db: SqliteConnection, cause: Cause.Cause<E>) =>
  db.execute("ROLLBACK").pipe(
    Effect.catchAllCause((rollbackCause) => Effect.failCause(Cause.sequential(cause, rollbackCause))),
    Effect.zipRight(Effect.failCause(cause))
  )

/** Runs `body` in one `BEGIN IMMEDIATE` transaction: committed if it succeeds, rolled back if not. */
export const transaction = <A, E>(
  db: SqliteConnection,
  body: Effect.Effect<A, E>
): Effect.Effect<A, E | SqliteDriverError> =>
  Effect.uninterruptible(
    Effect.gen(function* () {
      yield* db.execute("BEGIN IMMEDIATE")
      const result = yield* Effect.exit(body)
      if (Exit.isFailure(result)) return yield* rollback(db, result.cause)
      const committed = yield* Effect.exit(db.execute("COMMIT"))
      if (Exit.isFailure(committed)) return yield* rollback(db, committed.cause)
      return result.value
    })
  )
