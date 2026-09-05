import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { Cause, Deferred, Effect, Exit, Fiber, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { connectionTransaction, ConnectionTransaction } from "./transaction"
import { writeFileAtomic, removeConfigurationFile } from "./configuration-file"
import { withConnectionLock } from "./lock"

describe("connection compensation", () => {
  it("keeps compensation deadlines effective and continues after a timeout", async () => {
    let restored = false
    const exit = await Effect.runPromiseExit(connectionTransaction(Effect.gen(function* () {
      const tx = yield* ConnectionTransaction
      yield* tx.compensate("sibling", Effect.sync(() => { restored = true }))
      yield* tx.compensate("bounded undo", Effect.never.pipe(Effect.timeout("20 millis")))
      return yield* Effect.fail("original failure")
    })))
    expect(restored).toBe(true)
    expect(Exit.isFailure(exit) && Cause.pretty(exit.cause)).toContain("TimeoutException")
  }, 1000)
  it("runs every registered undo in reverse order and reports all failures alongside the original", async () => {
    const order: number[] = []
    const exit = await Effect.runPromiseExit(connectionTransaction(Effect.gen(function* () {
      const tx = yield* ConnectionTransaction
      for (const id of [1, 2, 3]) yield* tx.compensate(`undo ${id}`, Effect.sync(() => { order.push(id) }).pipe(Effect.zipRight(id === 2 ? Effect.die("undo defect") : id === 3 ? Effect.fail("undo failure") : Effect.void)))
      return yield* Effect.fail("original failure")
    })))
    expect(order).toEqual([3, 2, 1])
    expect(Exit.isFailure(exit) && Cause.pretty(exit.cause)).toContain("original failure")
    expect(Exit.isFailure(exit) && Cause.pretty(exit.cause)).toContain("undo failure")
    expect(Exit.isFailure(exit) && Cause.pretty(exit.cause)).toContain("undo defect")
  })

  it("restores interruption after mutation but does not roll back a committed transaction", async () => {
    const entered = Effect.runSync(Deferred.make<void>())
    let restored = false
    const fiber = Effect.runFork(connectionTransaction(Effect.gen(function* () {
      const tx = yield* ConnectionTransaction
      yield* tx.compensate("restore", Effect.sync(() => { restored = true }))
      yield* Deferred.succeed(entered, undefined)
      return yield* Effect.never
    })))
    await Effect.runPromise(Deferred.await(entered))
    await Effect.runPromise(Fiber.interrupt(fiber))
    expect(restored).toBe(true)
    restored = false
    await Effect.runPromiseExit(connectionTransaction(Effect.gen(function* () {
      const tx = yield* ConnectionTransaction
      yield* tx.compensate("restore", Effect.sync(() => { restored = true }))
      yield* tx.commit(Effect.void)
      return yield* Effect.fail("after commit")
    })))
    expect(restored).toBe(false)
  })

  it("restores changed and removed files, while preserving concurrent edits and still restoring siblings", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "connection-recovery-" })
      const first = `${root}/first`, second = `${root}/second`, removed = `${root}/removed`
      for (const file of [first, second, removed]) yield* fs.writeFileString(file, "before")
      const exit = yield* Effect.exit(connectionTransaction(Effect.gen(function* () {
        yield* writeFileAtomic(first, "ours")
        yield* writeFileAtomic(second, "ours")
        yield* removeConfigurationFile(removed)
        yield* fs.writeFileString(second, "user edit")
        return yield* Effect.fail("fixture failure")
      })))
      expect(yield* fs.readFileString(first)).toBe("before")
      expect(yield* fs.readFileString(second)).toBe("user edit")
      expect(yield* fs.readFileString(removed)).toBe("before")
      expect(Exit.isFailure(exit) && Cause.pretty(exit.cause)).toContain("Preserved concurrently changed configuration")
    }).pipe(Effect.provide(BunContext.layer))))
  })
})

describe("cross-process connection locking", () => {
  const worker = new URL("./fixtures/lock-worker.ts", import.meta.url).pathname
  it("serializes independent writers without lost updates", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "connection-lock-" })
      const file = `${root}/counter`
      yield* fs.writeFileString(file, "0")
      const codes = yield* Effect.all(Array.from({ length: 3 }, () => Command.make("bun", worker, file, "increment").pipe(Command.exitCode)), { concurrency: "unbounded" })
      expect(codes).toEqual([0, 0, 0])
      expect(yield* fs.readFileString(file)).toBe("30")
    }).pipe(Effect.provide(BunContext.layer))))
  })

  it("allows cancellation while waiting, and releases ownership when a process dies", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "connection-lock-kill-" })
      const file = `${root}/state`
      const owner = yield* Command.make("bun", worker, file, "hold").pipe(Command.start)
      const ready = yield* owner.stdout.pipe(Stream.decodeText(), Stream.take(1), Stream.runCollect)
      expect(String(ready)).toContain("acquired")
      const waiter = yield* withConnectionLock(file, Effect.void).pipe(Effect.fork)
      yield* Effect.sleep("100 millis")
      yield* Fiber.interrupt(waiter).pipe(Effect.timeout("1 second"))
      yield* owner.kill("SIGKILL")
      const killed = yield* Effect.exit(owner.exitCode)
      expect(Exit.isFailure(killed) && Cause.pretty(killed.cause)).toContain("SIGKILL")
      yield* withConnectionLock(file, Effect.void).pipe(Effect.timeout("2 seconds"))
    }).pipe(Effect.provide(BunContext.layer))))
  })

  it("fails with a typed error naming the lock when a holder never finishes", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "connection-lock-bound-" })
      const file = `${root}/state`
      const owner = yield* Command.make("bun", worker, file, "hold").pipe(Command.start)
      const ready = yield* owner.stdout.pipe(Stream.decodeText(), Stream.take(1), Stream.runCollect)
      expect(String(ready)).toContain("acquired")
      const result = yield* Effect.either(withConnectionLock(file, Effect.void, "300 millis"))
      expect(result._tag === "Left" && result.left._tag).toBe("HarnessConnectionLockTimedOut")
      if (result._tag === "Left" && result.left._tag === "HarnessConnectionLockTimedOut") {
        expect(result.left.path).toBe(`${file}.lock.sqlite`)
      }
      yield* owner.kill("SIGKILL")
      yield* Effect.exit(owner.exitCode)
    }).pipe(Effect.provide(BunContext.layer))))
  })
})
