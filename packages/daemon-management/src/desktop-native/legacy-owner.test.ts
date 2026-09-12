import { Database } from "bun:sqlite"
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { BunSqliteDriverLayer, BunSqliteDriver } from "../bun"
import { readLegacyOwner } from "./legacy-owner"
import { SqliteDriver, SqliteDriverFailure } from "../sqlite-driver"

let root: string
let path: string
beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "magnitude-legacy-owner-"))
  await mkdir(join(root, "acn"))
  path = join(root, "acn/coordination.sqlite")
})
afterEach(() => rm(root, { recursive: true, force: true }))
const read = () => Effect.runPromise(readLegacyOwner(root).pipe(Effect.provide(BunSqliteDriverLayer)))
const fixture = (rows: readonly (readonly [number, number, string, number])[] = []) => {
  const database = new Database(path, { create: true })
  database.exec("CREATE TABLE owner (id INTEGER, pid INTEGER, process_start_identity TEXT, port INTEGER)")
  for (const row of rows) database.query("INSERT INTO owner VALUES (?, ?, ?, ?)").run(...row)
  database.close()
}

describe("legacy migration owner read", () => {
  it("does not create an absent ownership database", async () => {
    expect(Option.isNone(await read())).toBe(true)
    await expect(readFile(path)).rejects.toMatchObject({ code: "ENOENT" })
  })
  it("reads an empty owner table without initializing ownership", async () => {
    fixture()
    const before = await readFile(path)
    expect(Option.isNone(await read())).toBe(true)
    expect(await readFile(path)).toEqual(before)
  })
  it("preserves an exact legacy identity as data without interpreting liveness", async () => {
    fixture([[1, 1234, "legacy-start-identity", 45678]])
    const before = await readFile(path)
    expect(Option.getOrThrow(await read())).toEqual({ pid: 1234, processStartIdentity: "legacy-start-identity", port: 45678 })
    expect(await readFile(path)).toEqual(before)
  })
  it.each([
    [[1, 0, "identity", 1234]],
    [[1, 1234, "", 1234]],
    [[1, 1234, "identity", 65536]],
    [[2, 1234, "identity", 1234]],
    [[1, 1234, "identity", 1234], [1, 1235, "other", 1235]],
  ] as const)("rejects malformed or multiple owner rows %#", async (...rows) => {
    fixture(rows)
    await expect(read()).rejects.toThrow("invalid owner record")
  })
  it("does not repair a corrupt database", async () => {
    await writeFile(path, "not a database")
    await expect(read()).rejects.toThrow()
    expect(await readFile(path, "utf8")).toBe("not a database")
  })
  it("does not turn a missing owner table into successful absence", async () => {
    const database = new Database(path, { create: true })
    database.exec("CREATE TABLE unrelated (value TEXT)")
    database.close()
    await expect(read()).rejects.toThrow("no such table")
  })
  it("preserves an open failure instead of treating the record as absent", async () => {
    fixture()
    const denied = SqliteDriver.of({ open: () => Effect.fail(new SqliteDriverFailure({ message: "access denied" })) })
    await expect(Effect.runPromise(readLegacyOwner(root).pipe(Effect.provideService(SqliteDriver, denied)))).rejects.toThrow("access denied")
  })
  it("rejects a symbolic link instead of following it", async () => {
    const target = join(root, "unrelated")
    await writeFile(target, "unrelated")
    await symlink(target, path)
    await expect(read()).rejects.toThrow("regular file owned by this user")
    expect(await readFile(target, "utf8")).toBe("unrelated")
  })
  it("enforces read-only mode in the SQLite adapter itself", async () => {
    fixture()
    await expect(Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const database = yield* BunSqliteDriver.open(path, { create: false, readOnly: true })
      yield* database.execute("INSERT INTO owner VALUES (1, 1234, 'identity', 1234)")
    })))).rejects.toThrow("readonly")
    expect(Option.isNone(await read())).toBe(true)
  })
})
