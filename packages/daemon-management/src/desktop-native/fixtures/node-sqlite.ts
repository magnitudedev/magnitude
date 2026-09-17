import { strict as assert } from "node:assert"
import { existsSync, mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { Effect, Either } from "effect"
import { NodeSqliteDriver } from "../../node-sqlite-driver"

const root = mkdtempSync(`${tmpdir()}/magnitude-node-sqlite-`)
const path = `${root}/database ?# ü.sqlite`
try {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const absent = yield* NodeSqliteDriver.open(path, { create: false }).pipe(Effect.either)
    assert(Either.isLeft(absent))
    assert(!existsSync(path))
    const first = yield* NodeSqliteDriver.open(path, { create: true })
    yield* first.execute("CREATE TABLE entries (text TEXT, enabled INTEGER, bytes BLOB)")
    yield* first.execute("INSERT INTO entries VALUES (?, ?, ?)", ["quoted ' value", true, new Uint8Array([1, 2])])
    const rows = yield* first.query("SELECT text, enabled, hex(bytes) AS bytes FROM entries")
    assert.deepEqual(JSON.parse(JSON.stringify(rows)), [{ text: "quoted ' value", enabled: 1, bytes: "0102" }])
    const second = yield* NodeSqliteDriver.open(path, { create: false })
    yield* first.execute("BEGIN IMMEDIATE")
    const busy = yield* second.execute("BEGIN IMMEDIATE").pipe(Effect.either)
    assert(Either.isLeft(busy) && busy.left._tag === "SqliteDriverBusy")
    yield* first.execute("ROLLBACK")
    yield* second.execute("BEGIN IMMEDIATE")
    yield* second.execute("ROLLBACK")
  })))
  // A new scope can reopen and write after both handles have finalized.
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const reopened = yield* NodeSqliteDriver.open(path, { create: false })
    yield* reopened.execute("BEGIN EXCLUSIVE")
    yield* reopened.execute("COMMIT")
  })))
  console.log("Node SQLite acceptance passed")
} finally {
  rmSync(root, { recursive: true })
}
