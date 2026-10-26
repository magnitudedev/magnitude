import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { Cause, Effect, Exit } from "effect"
import { BunSqliteDriver } from "./bun"
import { enableWal, SqliteCheckpointIncomplete, SqliteWalRefused, transaction, truncateWal } from "./database"
import { type SqliteConnection, SqliteDriverFailure } from "./driver"

let root: string

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "magnitude-sqlite-database-"))
})

afterEach(async () => {
  await rm(root, { recursive: true, force: true })
})

const run = <A, E>(body: Effect.Effect<A, E, never>) => Effect.runPromiseExit(body)
const open = (name = "db.sqlite") => BunSqliteDriver.open(join(root, name), { create: true })

// A connection whose listed statements fail, recording every statement it runs.
const scripted = (failing: ReadonlyArray<string>) => {
  const statements: string[] = []
  const db: SqliteConnection = {
    execute: (sql) => Effect.suspend(() => {
      statements.push(sql)
      return failing.includes(sql) ? Effect.fail(new SqliteDriverFailure({ message: `${sql} failed` })) : Effect.void
    }),
    query: (sql) => Effect.dieMessage(`unexpected query ${sql}`),
  }
  return { db, statements }
}

describe("transaction", () => {
  it("rolls back a transaction whose COMMIT fails, and leaves the connection usable", async () => {
    const exit = await run(Effect.scoped(Effect.gen(function* () {
      const db = yield* open()
      yield* db.execute("PRAGMA foreign_keys=ON")
      yield* db.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
      yield* db.execute("CREATE TABLE child (parent INTEGER REFERENCES parent (id) DEFERRABLE INITIALLY DEFERRED)")
      // A deferred foreign key is checked by COMMIT, which fails and leaves the transaction open.
      const failed = yield* Effect.exit(transaction(db, db.execute("INSERT INTO child VALUES (1)")))
      const rows = yield* db.query("SELECT COUNT(*) AS n FROM child")
      yield* transaction(db, db.execute("INSERT INTO parent VALUES (1)"))
      return { failed, rows }
    })))
    const { failed, rows } = Exit.getOrElse(exit, (cause) => { throw Cause.squash(cause) })
    expect(Exit.isFailure(failed) && [...Cause.failures(failed.cause)].map((error) => error._tag)).toEqual(["SqliteDriverFailure"])
    expect(rows).toEqual([{ n: 0 }])
  })

  it("keeps a failed COMMIT's failure and its ROLLBACK's", async () => {
    const { db, statements } = scripted(["COMMIT", "ROLLBACK"])
    const exit = await run(transaction(db, Effect.succeed(1)))
    expect(statements).toEqual(["BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"])
    expect(Exit.isFailure(exit) && [...Cause.failures(exit.cause)]).toEqual([
      new SqliteDriverFailure({ message: "COMMIT failed" }),
      new SqliteDriverFailure({ message: "ROLLBACK failed" }),
    ])
  })

  it("keeps the body's failure and a failing ROLLBACK's", async () => {
    const { db, statements } = scripted(["ROLLBACK"])
    const exit = await run(transaction(db, Effect.fail("body failed" as const)))
    expect(statements).toEqual(["BEGIN IMMEDIATE", "ROLLBACK"])
    expect(Exit.isFailure(exit) && [...Cause.failures(exit.cause)]).toEqual([
      "body failed",
      new SqliteDriverFailure({ message: "ROLLBACK failed" }),
    ])
  })
})

describe("write-ahead log", () => {
  it("fails when SQLite keeps another journal mode", async () => {
    const exit = await run(Effect.scoped(Effect.flatMap(BunSqliteDriver.open(":memory:", { create: true }), enableWal)))
    expect(exit).toEqual(Exit.fail(new SqliteWalRefused({ journalMode: "memory" })))
  })

  it("fails a truncation that a reader blocks, and completes one that nothing blocks", async () => {
    const exit = await run(Effect.scoped(Effect.gen(function* () {
      const writer = yield* open()
      const reader = yield* open()
      yield* enableWal(writer)
      yield* writer.execute("CREATE TABLE t (x)")
      yield* writer.execute("INSERT INTO t VALUES (1)")
      yield* reader.execute("BEGIN")
      yield* reader.query("SELECT * FROM t")
      yield* writer.execute("INSERT INTO t VALUES (2)")
      const blocked = yield* Effect.flip(truncateWal(writer))
      yield* reader.execute("COMMIT")
      yield* truncateWal(writer)
      return blocked
    })))
    expect(Exit.isSuccess(exit) && exit.value).toEqual(expect.any(SqliteCheckpointIncomplete))
    expect(Exit.isSuccess(exit) && exit.value).toMatchObject({ busy: 1 })
  })
})
