import { Deferred, Effect, Exit, Queue, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { materializeProjection } from "./materialized-projection"

describe("materializeProjection", () => {
  it("coalesces invalidations and publishes only changed results", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const invalidations = yield* Queue.unbounded<void>()
      const reads = yield* Ref.make(0)
      const value = yield* Ref.make(0)
      const secondReadStarted = yield* Deferred.make<void>()
      const releaseSecondRead = yield* Deferred.make<void>()
      const project = Ref.updateAndGet(reads, (count) => count + 1).pipe(
        Effect.tap((count) => count === 2
          ? Deferred.succeed(secondReadStarted, undefined).pipe(
              Effect.zipRight(Deferred.await(releaseSecondRead)),
            )
          : Effect.void),
        Effect.zipRight(Ref.get(value)),
      )
      const projection = yield* materializeProjection({
        project,
        invalidations: Stream.fromQueue(invalidations),
        equivalent: (left, right) => left === right,
      })
      const published = yield* projection.changes.pipe(
        Stream.take(1),
        Stream.runCollect,
        Effect.fork,
      )
      yield* Effect.yieldNow()

      yield* Queue.offer(invalidations, undefined)
      yield* Deferred.await(secondReadStarted)
      yield* Ref.set(value, 1)
      yield* Effect.forEach(
        Array.from({ length: 100 }),
        () => Queue.offer(invalidations, undefined),
        { discard: true },
      )
      yield* Deferred.succeed(releaseSecondRead, undefined)
      const publishedValues = [...(yield* Effect.fromFiber(published))]

      return {
        current: yield* projection.get,
        published: publishedValues,
        reads: yield* Ref.get(reads),
      }
    })))

    expect(result).toEqual({ current: 1, published: [1], reads: 3 })
  })

  it("surfaces a projection defect and can recover on a later invalidation", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const invalidations = yield* Queue.unbounded<void>()
      const reads = yield* Ref.make(0)
      const failed = yield* Deferred.make<void>()
      const recovered = yield* Deferred.make<void>()
      const projection = yield* materializeProjection({
        project: Ref.updateAndGet(reads, (count) => count + 1).pipe(Effect.flatMap((count) => {
          if (count === 2) {
            return Deferred.succeed(failed, undefined).pipe(
              Effect.zipRight(Effect.dieMessage("invalid projection")),
            )
          }
          if (count === 3) return Deferred.succeed(recovered, undefined).pipe(Effect.as(1))
          return Effect.succeed(0)
        })),
        invalidations: Stream.fromQueue(invalidations),
        equivalent: (left, right) => left === right,
      })

      yield* Queue.offer(invalidations, undefined)
      yield* Deferred.await(failed)
      yield* Effect.yieldNow()
      const failedRead = yield* projection.get.pipe(Effect.exit)
      yield* Queue.offer(invalidations, undefined)
      yield* Deferred.await(recovered)
      yield* Effect.yieldNow()
      return { failedRead, recovered: yield* projection.get }
    })))

    expect(Exit.isFailure(result.failedRead)).toBe(true)
    expect(result.recovered).toBe(1)
  })
})
