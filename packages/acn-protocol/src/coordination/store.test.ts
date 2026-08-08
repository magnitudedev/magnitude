import { mkdir, mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { BunContext } from "@effect/platform-bun"
import { Database } from "bun:sqlite"
import { Effect, Layer, Option } from "effect"
import { afterEach, beforeEach, describe, expect, test } from "vitest"
import { ProcessStartIdentitySchema } from "../acn-identity"
import { BunSqliteDriverLayer } from "./bun"
import { makeAcnOwnerStore } from "./owner-store"

const platform = Layer.merge(BunContext.layer, BunSqliteDriverLayer)

const owner = (name: string, port: number) => ({
  pid: process.pid,
  processStartIdentity: ProcessStartIdentitySchema.make(`test:${name}`),
  port,
})

describe("ACN coordination database", () => {
  let root: string

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-acn-coordination-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  test("persists only exact owner coordination state", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const owners = yield* makeAcnOwnerStore(root)
      expect(Option.isNone(yield* owners.current)).toBe(true)
    }).pipe(Effect.provide(platform)))

    const database = new Database(join(root, "acn", "coordination.sqlite"))
    const tables = database.query(
      "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
    ).all() as Array<{ readonly name: string }>
    database.close()
    expect(tables).toEqual([{ name: "owner" }])
  })

  test("atomically replaces only the expected owner", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const owners = yield* makeAcnOwnerStore(root)
      const first = owner("first", 42_001)
      const second = owner("second", 42_002)

      expect(yield* owners.replaceOwner(Option.none(), first)).toEqual({
        _tag: "Replaced",
      })
      expect(yield* owners.replaceOwner(Option.none(), second)).toEqual({
        _tag: "OwnerChanged",
        owner: Option.some(first),
      })
      expect(yield* owners.current).toEqual(Option.some(first))
      expect(yield* owners.replaceOwner(Option.some(first), second)).toEqual({
        _tag: "Replaced",
      })
      expect(yield* owners.current).toEqual(Option.some(second))
    }).pipe(Effect.provide(platform)))
  })

  test("admits exactly one candidate for one expected owner", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const owners = yield* makeAcnOwnerStore(root)
      return yield* Effect.all([
        owners.replaceOwner(Option.none(), owner("one", 42_001)),
        owners.replaceOwner(Option.none(), owner("two", 42_002)),
      ], { concurrency: "unbounded" })
    }).pipe(Effect.provide(platform)))

    expect(result.filter((value) => value._tag === "Replaced")).toHaveLength(1)
    expect(result.filter((value) => value._tag === "OwnerChanged")).toHaveLength(1)
  })

  test("fails typed instead of treating an incompatible owner table as absence", async () => {
    const directory = join(root, "acn")
    await mkdir(directory, { recursive: true })
    const database = new Database(join(directory, "coordination.sqlite"), { create: true })
    database.query(`CREATE TABLE owner (
      id INTEGER,
      pid INTEGER,
      process_start_identity TEXT,
      port INTEGER
    )`).run()
    database.query("INSERT INTO owner VALUES (1, 1, 'first', 42001)").run()
    database.query("INSERT INTO owner VALUES (2, 2, 'second', 42002)").run()
    database.close()

    const result = await Effect.runPromise(Effect.gen(function* () {
      const owners = yield* makeAcnOwnerStore(root)
      return yield* Effect.either(owners.current)
    }).pipe(Effect.provide(platform)))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("AcnProcessStoreInvalid")
  })
})
