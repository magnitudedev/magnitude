import { mkdir, mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { ProcessStartIdentitySchema } from "../acn-identity"
import { AcnRevisionSchema } from "../acn-revision"
import { Effect, Layer, Option } from "effect"
import { afterEach, beforeEach, describe, expect, test } from "vitest"
import { makeAcnOwnerLock } from "./owner-lock"
import { makeAcnRevisionStore } from "./revision-store"
import { BunSqliteMutexLayer } from "./bun"

const platform = Layer.mergeAll(BunFileSystem.layer, BunPath.layer, BunSqliteMutexLayer)

describe("ACN process stores", () => {
  let root: string

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-acn-coordination-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  test("selects published markers permanently and development markers only while held", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const store = yield* makeAcnRevisionStore(root)
      const published = AcnRevisionSchema.make(1_000_000)
      const development = AcnRevisionSchema.make(1_000_001)
      yield* store.registerPublished(published)
      const before = yield* store.selected
      const hold = yield* store.holdDevelopment(development, "0123456789abcdef")
      const active = yield* store.selected
      yield* hold.close
      const after = yield* store.selected
      return { before, active, after }
    }).pipe(Effect.provide(platform))))

    expect(Option.getOrUndefined(result.before)).toBe(1_000_000)
    expect(Option.getOrUndefined(result.active)).toBe(1_000_001)
    expect(Option.getOrUndefined(result.after)).toBe(1_000_000)
  })

  test("admits one owner and observes metadata only while that owner holds the lock", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const first = yield* makeAcnOwnerLock(root)
      const second = yield* makeAcnOwnerLock(root)
      const acquired = yield* first.tryAcquire
      expect(Option.isSome(acquired)).toBe(true)
      if (Option.isNone(acquired)) return
      expect(yield* second.observe).toEqual({ _tag: "Publishing" })
      yield* acquired.value.publish({
        pid: process.pid,
        processStartIdentity: ProcessStartIdentitySchema.make("test:owner"),
        port: 42_001,
      })
      expect(Option.isNone(yield* second.tryAcquire)).toBe(true)
      expect(yield* second.observe).toEqual({
        _tag: "Locked",
        owner: {
          pid: process.pid,
          processStartIdentity: "test:owner",
          port: 42_001,
        },
      })
      yield* acquired.value.close
      expect(yield* second.observe).toEqual({ _tag: "Unlocked" })
    }).pipe(Effect.provide(platform))))
  })

  test("fails closed on malformed relevant revision markers", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const store = yield* makeAcnRevisionStore(root)
      const revision = AcnRevisionSchema.make(1_000_000)
      yield* store.registerPublished(revision)
      yield* Effect.tryPromise(() => Bun.write(
        join(root, "acn", "revisions", "00000000001000000001"),
        "not-a-development-key",
      ))
      return yield* Effect.either(store.selected)
    }).pipe(Effect.provide(platform))))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("AcnProcessStoreInvalid")
  })

  test("does not let two development identities claim one revision", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const store = yield* makeAcnRevisionStore(root)
      const revision = AcnRevisionSchema.make(1_000_001)
      const hold = yield* store.holdDevelopment(revision, "0123456789abcdef")
      const conflict = yield* Effect.either(
        store.holdDevelopment(revision, "fedcba9876543210"),
      )
      yield* hold.close
      return conflict
    }).pipe(Effect.provide(platform))))

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("AcnProcessStoreInvalid")
  })

  test("keeps a development revision active until its last holder closes", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const store = yield* makeAcnRevisionStore(root)
      const revision = AcnRevisionSchema.make(1_000_001)
      const first = yield* store.holdDevelopment(revision, "0123456789abcdef")
      const second = yield* store.holdDevelopment(revision, "0123456789abcdef")
      yield* first.close
      const oneRemaining = yield* store.selected
      yield* second.close
      const noneRemaining = yield* store.selected
      return { oneRemaining, noneRemaining }
    }).pipe(Effect.provide(platform))))

    expect(Option.getOrUndefined(result.oneRemaining)).toBe(1_000_001)
    expect(Option.isNone(result.noneRemaining)).toBe(true)
  })

  test("ignores malformed unlocked owner bytes when acquiring new authority", async () => {
    await mkdir(join(root, "acn"), { recursive: true })
    await Bun.write(join(root, "acn", "owner.json"), "partial-owner-publication")
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const lock = yield* makeAcnOwnerLock(root)
      const owner = Option.getOrThrow(yield* lock.tryAcquire)
      expect(Option.isNone(owner.previous)).toBe(true)
    }).pipe(Effect.provide(platform))))
  })

  test("preserves predecessor evidence when a candidate releases before publishing", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const lock = yield* makeAcnOwnerLock(root)
      const first = yield* lock.tryAcquire
      expect(Option.isSome(first)).toBe(true)
      if (Option.isNone(first)) return
      const predecessor = {
        pid: process.pid,
        processStartIdentity: ProcessStartIdentitySchema.make("test:predecessor"),
        port: 42_001,
      }
      yield* first.value.publish(predecessor)
      yield* first.value.close

      const interrupted = yield* lock.tryAcquire
      expect(Option.isSome(interrupted)).toBe(true)
      if (Option.isNone(interrupted)) return
      expect(Option.getOrUndefined(interrupted.value.previous)).toEqual(predecessor)
      yield* interrupted.value.close

      const successor = yield* lock.tryAcquire
      expect(Option.isSome(successor)).toBe(true)
      if (Option.isSome(successor)) {
        expect(Option.getOrUndefined(successor.value.previous)).toEqual(predecessor)
      }
    }).pipe(Effect.provide(platform))))
  })
})
