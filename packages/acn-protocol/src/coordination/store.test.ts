import { mkdir, mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { BunContext } from "@effect/platform-bun"
import { Database } from "bun:sqlite"
import { Effect, Layer, Option } from "effect"
import { afterEach, beforeEach, describe, expect, test } from "vitest"
import { ProcessStartIdentitySchema } from "../acn-identity"
import { AcnRevisionSchema } from "../acn-revision"
import { BunSqliteDriverLayer } from "./bun"
import { makeAcnOwnerStore } from "./owner-store"
import { makeAcnRevisionStore } from "./revision-store"

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

  test("stores every revision permanently and selects the greatest scalar", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const store = yield* makeAcnRevisionStore(root)
      const first = AcnRevisionSchema.make(1_000_001)
      const second = AcnRevisionSchema.make(1_000_002)
      expect(Option.isNone(yield* store.selected)).toBe(true)
      yield* store.register(second)
      yield* store.register(first)
      yield* store.register(second)
      return yield* store.selected
    }).pipe(Effect.provide(platform)))

    expect(Option.getOrUndefined(result)).toBe(1_000_002)
  })

  test("atomically replaces only the expected owner at the selected revision", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const revisions = yield* makeAcnRevisionStore(root)
      const owners = yield* makeAcnOwnerStore(root)
      const revision = AcnRevisionSchema.make(1_000_001)
      const first = owner("first", 42_001)
      const second = owner("second", 42_002)

      yield* revisions.register(revision)
      expect(yield* owners.replaceOwner(Option.none(), first, revision)).toEqual({
        _tag: "Replaced",
      })
      expect(yield* owners.replaceOwner(Option.none(), second, revision)).toEqual({
        _tag: "OwnerChanged",
        owner: Option.some(first),
      })
      expect(yield* owners.current).toEqual(Option.some(first))
      expect(yield* owners.replaceOwner(Option.some(first), second, revision)).toEqual({
        _tag: "Replaced",
      })
      expect(yield* owners.current).toEqual(Option.some(second))
    }).pipe(Effect.provide(platform)))
  })

  test("rejects admission when selection changes and preserves predecessor evidence", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const revisions = yield* makeAcnRevisionStore(root)
      const owners = yield* makeAcnOwnerStore(root)
      const oldRevision = AcnRevisionSchema.make(1_000_001)
      const newRevision = AcnRevisionSchema.make(1_000_002)
      const predecessor = owner("predecessor", 42_001)
      const candidate = owner("candidate", 42_002)

      yield* revisions.register(oldRevision)
      yield* owners.replaceOwner(Option.none(), predecessor, oldRevision)
      yield* revisions.register(newRevision)
      expect(yield* owners.replaceOwner(
        Option.some(predecessor),
        candidate,
        oldRevision,
      )).toEqual({
        _tag: "SelectionChanged",
        revision: Option.some(newRevision),
      })
      expect(yield* owners.current).toEqual(Option.some(predecessor))
    }).pipe(Effect.provide(platform)))
  })

  test("admits exactly one candidate for one expected owner", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const revisions = yield* makeAcnRevisionStore(root)
      const owners = yield* makeAcnOwnerStore(root)
      const revision = AcnRevisionSchema.make(1_000_001)
      yield* revisions.register(revision)
      return yield* Effect.all([
        owners.replaceOwner(Option.none(), owner("one", 42_001), revision),
        owners.replaceOwner(Option.none(), owner("two", 42_002), revision),
      ], { concurrency: "unbounded" })
    }).pipe(Effect.provide(platform)))

    expect(result.filter((value) => value._tag === "Replaced")).toHaveLength(1)
    expect(result.filter((value) => value._tag === "OwnerChanged")).toHaveLength(1)
  })

  test("fails typed instead of treating an incompatible owner table as absence", async () => {
    const directory = join(root, "acn")
    await mkdir(directory, { recursive: true })
    const database = new Database(join(directory, "coordination.sqlite"), { create: true })
    database.query("CREATE TABLE revisions (revision INTEGER PRIMARY KEY)").run()
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
