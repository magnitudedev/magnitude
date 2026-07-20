import { Effect, Fiber, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { AcnActivityTracker, AcnActivityTrackerLive } from "./activity-tracker"

describe("AcnActivityTracker", () => {
  it("marks command activity", async () => {
    const program = Effect.gen(function* () {
      const activity = yield* AcnActivityTracker
      yield* activity.markCommand("SendMessage")

      return yield* activity.current
    }).pipe(Effect.provide(AcnActivityTrackerLive))

    const result = await Effect.runPromise(program)

    expect(result.lastCommandAt).toBeGreaterThan(0)
    expect(result.lastActivityAt).toBeGreaterThan(0)
  })

  it("emits changes when activity state changes", async () => {
    const program = Effect.gen(function* () {
      const activity = yield* AcnActivityTracker
      const fiber = yield* activity.changes.pipe(
        Stream.take(1),
        Stream.runCollect,
        Effect.fork,
      )

      yield* Effect.sleep("10 millis")
      yield* activity.markCommand("SendMessage")

      return yield* Fiber.join(fiber)
    }).pipe(Effect.provide(AcnActivityTrackerLive))

    const result = await Effect.runPromise(program)

    expect(result.length).toBe(1)
  })

  it("holds active work for the full scoped operation", async () => {
    const program = Effect.gen(function* () {
      const activity = yield* AcnActivityTracker
      const latch = yield* Effect.makeLatch()
      const fiber = yield* activity.withActiveWork("download", latch.await).pipe(Effect.fork)
      yield* Effect.yieldNow()
      const during = yield* activity.hasActiveWork
      yield* latch.open
      yield* Fiber.join(fiber)
      const after = yield* activity.hasActiveWork
      return { during, after }
    }).pipe(Effect.provide(AcnActivityTrackerLive))

    await expect(Effect.runPromise(program)).resolves.toEqual({ during: true, after: false })
  })
})
