import { Deferred, Effect, Fiber, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { makeClientInvalidations } from "./client-invalidations"

describe("ClientInvalidations", () => {
  it("broadcasts each event to independent consumers", async () => {
    const program = Effect.gen(function* () {
      const invalidations = yield* makeClientInvalidations
      const firstReady = yield* Deferred.make<void>()
      const secondReady = yield* Deferred.make<void>()
      const first = yield* invalidations.events.pipe(
        Stream.tap(() => Deferred.succeed(firstReady, undefined)),
        Stream.take(1),
        Stream.runCollect,
        Effect.fork,
      )
      const second = yield* invalidations.events.pipe(
        Stream.tap(() => Deferred.succeed(secondReady, undefined)),
        Stream.take(1),
        Stream.runCollect,
        Effect.fork,
      )
      yield* Effect.yieldNow()
      yield* invalidations.publish({ _tag: "Projects" })
      yield* Deferred.await(firstReady)
      yield* Deferred.await(secondReady)
      return [yield* Fiber.join(first), yield* Fiber.join(second)] as const
    })

    const [first, second] = await Effect.runPromise(program)
    expect(Array.from(first)).toEqual([{ _tag: "Projects" }])
    expect(Array.from(second)).toEqual([{ _tag: "Projects" }])
  })
})
