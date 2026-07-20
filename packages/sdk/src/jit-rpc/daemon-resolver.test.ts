import { describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Option, Ref } from "effect"
import { makeJitDaemonCoordinator, type JitDaemonProvider } from "./index"

describe("makeJitDaemonCoordinator", () => {
  it("discovers before spawning and caches the endpoint", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const discoverCalls = yield* Ref.make(0)
      const spawnCalls = yield* Ref.make(0)
      const provider: JitDaemonProvider<never> = {
        discover: () =>
          Ref.updateAndGet(discoverCalls, (count) => count + 1).pipe(
            Effect.map(() => Option.some({ url: "http://daemon" })),
          ),
        spawn: () =>
          Ref.update(spawnCalls, (count) => count + 1).pipe(
            Effect.as({ url: "http://spawned" }),
          ),
      }
      const coordinator = yield* makeJitDaemonCoordinator(provider)

      expect(yield* Ref.get(discoverCalls)).toBe(0)
      expect(yield* Ref.get(spawnCalls)).toBe(0)

      const first = yield* coordinator.ensure
      const second = yield* coordinator.ensure

      expect(first.endpoint).toEqual({ url: "http://daemon" })
      expect(second).toEqual(first)
      expect(yield* Ref.get(discoverCalls)).toBe(1)
      expect(yield* Ref.get(spawnCalls)).toBe(0)
    }))
  })

  it("single-flights concurrent spawn and invalidates by endpoint", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const spawnCalls = yield* Ref.make(0)
      const entered = yield* Deferred.make<void>()
      const continueSpawn = yield* Deferred.make<void>()
      const provider: JitDaemonProvider<never> = {
        discover: () => Effect.succeed(Option.none()),
        spawn: () =>
          Effect.gen(function* () {
            yield* Ref.update(spawnCalls, (count) => count + 1)
            yield* Deferred.succeed(entered, undefined)
            yield* Deferred.await(continueSpawn)
            return { url: "http://spawned" }
          }),
      }
      const coordinator = yield* makeJitDaemonCoordinator(provider)

      const first = yield* Effect.fork(coordinator.ensure)
      yield* Deferred.await(entered)
      const second = yield* Effect.fork(coordinator.ensure)
      yield* Deferred.succeed(continueSpawn, undefined)

      const firstLease = yield* Fiber.join(first)
      expect(firstLease.endpoint).toEqual({ url: "http://spawned" })
      expect(yield* Fiber.join(second)).toEqual(firstLease)
      expect(yield* Ref.get(spawnCalls)).toBe(1)

      yield* coordinator.invalidate(firstLease)
      const restarted = yield* Effect.fork(coordinator.ensure)
      yield* Deferred.succeed(continueSpawn, undefined)
      const secondLease = yield* Fiber.join(restarted)
      expect(secondLease.endpoint).toEqual({ url: "http://spawned" })
      expect(secondLease.generation).not.toBe(firstLease.generation)
      expect(yield* Ref.get(spawnCalls)).toBe(2)

      // A delayed failure from the first attempt must not invalidate a newer
      // resolution, even when the daemon reused the same URL.
      yield* coordinator.invalidate(firstLease)
      expect(yield* coordinator.ensure).toEqual(secondLease)
      expect(yield* Ref.get(spawnCalls)).toBe(2)
    }))
  })

  it("shares one coordinator across independent consumers", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const discoverCalls = yield* Ref.make(0)
      const spawnCalls = yield* Ref.make(0)
      const entered = yield* Deferred.make<void>()
      const continueSpawn = yield* Deferred.make<void>()
      const provider: JitDaemonProvider<never> = {
        discover: () => Ref.update(discoverCalls, (count) => count + 1).pipe(Effect.as(Option.none())),
        spawn: () => Effect.gen(function* () {
          yield* Ref.update(spawnCalls, (count) => count + 1)
          yield* Deferred.succeed(entered, undefined)
          yield* Deferred.await(continueSpawn)
          return { url: "http://shared" }
        }),
      }
      const coordinator = yield* makeJitDaemonCoordinator(provider)

      const consumers = yield* Effect.forEach(
        Array.from({ length: 100 }),
        () => coordinator.ensure,
        { concurrency: "unbounded" },
      ).pipe(Effect.fork)
      yield* Deferred.await(entered)
      yield* Deferred.succeed(continueSpawn, undefined)

      const leases = yield* Fiber.join(consumers)
      expect(leases.map((lease) => lease.endpoint)).toEqual(
        Array.from({ length: 100 }, () => ({ url: "http://shared" })),
      )
      expect(new Set(leases.map((lease) => lease.generation)).size).toBe(1)
      expect(yield* Ref.get(discoverCalls)).toBe(1)
      expect(yield* Ref.get(spawnCalls)).toBe(1)
    }))
  })
})
