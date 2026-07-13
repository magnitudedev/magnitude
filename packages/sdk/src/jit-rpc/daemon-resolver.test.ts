import { describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Option, Ref } from "effect"
import { makeJitDaemonResolver, type JitDaemonProvider } from "./index"

describe("makeJitDaemonResolver", () => {
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
      const resolver = makeJitDaemonResolver(provider)

      const first = yield* resolver.resolve
      const second = yield* resolver.resolve

      expect(first).toEqual({ url: "http://daemon" })
      expect(second).toEqual({ url: "http://daemon" })
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
      const resolver = makeJitDaemonResolver(provider)

      const first = yield* Effect.fork(resolver.resolve)
      yield* Deferred.await(entered)
      const second = yield* Effect.fork(resolver.resolve)
      yield* Deferred.succeed(continueSpawn, undefined)

      expect(yield* Fiber.join(first)).toEqual({ url: "http://spawned" })
      expect(yield* Fiber.join(second)).toEqual({ url: "http://spawned" })
      expect(yield* Ref.get(spawnCalls)).toBe(1)

      yield* resolver.invalidate({ url: "http://other" })
      expect(yield* resolver.resolve).toEqual({ url: "http://spawned" })
      expect(yield* Ref.get(spawnCalls)).toBe(1)

      yield* resolver.invalidate({ url: "http://spawned" })
      const restarted = yield* Effect.fork(resolver.resolve)
      yield* Deferred.succeed(continueSpawn, undefined)
      expect(yield* Fiber.join(restarted)).toEqual({ url: "http://spawned" })
      expect(yield* Ref.get(spawnCalls)).toBe(2)
    }))
  })
})
