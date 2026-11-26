import { Chunk, Deferred, Effect, Fiber, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { AcnChanges, AcnChangesLive } from "./changes"

describe("ACN changes", () => {
  it("serves published pokes to subscribers on one stream", async () => {
    const values = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const changes = yield* AcnChanges
      const collected = yield* changes.stream.pipe(Stream.take(2), Stream.runCollect, Effect.fork)
      yield* Effect.sleep("5 millis")
      yield* changes.publish({ operation: "GetModelSlots" })
      yield* changes.publish({ operation: "ListChats", key: { chatId: "c1" } })
      return Chunk.toReadonlyArray(yield* Fiber.join(collected))
    }).pipe(Effect.provide(AcnChangesLive))))

    expect(values).toEqual([
      { operation: "GetModelSlots" },
      { operation: "ListChats", key: { chatId: "c1" } },
    ])
  })

  it("broadens excessive keyed invalidations for a slow subscriber", async () => {
    const values = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const changes = yield* AcnChanges
      const firstObserved = yield* Deferred.make<void>()
      const releaseFirst = yield* Deferred.make<void>()
      const collected = yield* changes.stream.pipe(
        Stream.tap((change) => change.key === undefined
          ? Effect.void
          : Deferred.succeed(firstObserved, undefined).pipe(
              Effect.zipRight(Deferred.await(releaseFirst)),
            )),
        Stream.take(2),
        Stream.runCollect,
        Effect.fork,
      )
      yield* Effect.sleep("5 millis")
      yield* changes.publish({ operation: "ListChats", key: { chatId: "first" } })
      yield* Deferred.await(firstObserved)
      yield* Effect.forEach(
        Array.from({ length: 100 }, (_, index) => index),
        (index) => changes.publish({ operation: "ListChats", key: { chatId: `pending-${index}` } }),
        { discard: true },
      )
      yield* Deferred.succeed(releaseFirst, undefined)
      return Chunk.toReadonlyArray(yield* Fiber.join(collected))
    }).pipe(Effect.provide(AcnChangesLive))))

    expect(values).toEqual([
      { operation: "ListChats", key: { chatId: "first" } },
      { operation: "ListChats" },
    ])
  })
})
