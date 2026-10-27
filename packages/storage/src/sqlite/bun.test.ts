import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, beforeEach, expect, it } from "vitest"
import { Effect, Exit } from "effect"
import { BunSqliteDriver } from "./bun"
import { SqliteDriverNotADatabase } from "./driver"

let root: string

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "magnitude-sqlite-bun-"))
})

afterEach(async () => {
  await rm(root, { recursive: true, force: true })
})

it("reports a file that is not a SQLite database as such", async () => {
  const path = join(root, "text.sqlite")
  await writeFile(path, "plain text, not a database ".repeat(40))
  const exit = await Effect.runPromiseExit(Effect.scoped(Effect.flatMap(
    BunSqliteDriver.open(path, { create: false }),
    (db) => db.query("PRAGMA user_version"),
  )))
  expect(exit).toEqual(Exit.fail(new SqliteDriverNotADatabase()))
})
