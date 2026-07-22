import { Deferred, Effect, Fiber, Layer, Option, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { AcnShutdown, AcnShutdownLive } from "./acn-shutdown"
import { AcnActivityTracker, AcnActivityTrackerLive } from "./activity-tracker"

const TestLayer = AcnActivityTrackerLive("30 minutes").pipe(
  Layer.provideMerge(AcnShutdownLive),
  Layer.provideMerge(TestContext.TestContext),
)

describe("AcnActivityTracker", () => {
  it("requests shutdown at the exact idle deadline", async () => {
    const program = Effect.gen(function* () {
      const activity = yield* AcnActivityTracker
      const shutdown = yield* AcnShutdown
      yield* Effect.yieldNow()
      yield* TestClock.adjust("1799999 millis")
      expect(Option.isNone(yield* shutdown.current)).toBe(true)
      yield* TestClock.adjust("1 millis")
      expect((yield* shutdown.await).reason).toBe("idle")
      expect((yield* activity.current).phase).toBe("retired")
    }).pipe(Effect.provide(TestLayer))

    await Effect.runPromise(program)
  })

  it("protects in-flight demand and starts a full interval on release", async () => {
    const program = Effect.gen(function* () {
      const activity = yield* AcnActivityTracker
      const shutdown = yield* AcnShutdown
      const latch = yield* Deferred.make<void>()
      const fiber = yield* activity.withUse("work", Deferred.await(latch)).pipe(Effect.fork)
      yield* Effect.yieldNow()
      yield* TestClock.adjust("2 hours")
      expect(Option.isNone(yield* shutdown.current)).toBe(true)
      yield* Deferred.succeed(latch, undefined)
      yield* Fiber.join(fiber)
      yield* Effect.yieldNow()
      yield* TestClock.adjust("30 minutes")
      expect((yield* shutdown.await).reason).toBe("idle")
    }).pipe(Effect.provide(TestLayer))

    await Effect.runPromise(program)
  })

  it("starts the initial allowance at readiness rather than process construction", async () => {
    const layer = AcnActivityTrackerLive("30 minutes", false).pipe(
      Layer.provideMerge(AcnShutdownLive),
      Layer.provideMerge(TestContext.TestContext),
    )
    const program = Effect.gen(function* () {
      const activity = yield* AcnActivityTracker
      const shutdown = yield* AcnShutdown
      yield* TestClock.adjust("2 hours")
      expect(Option.isNone(yield* shutdown.current)).toBe(true)
      yield* activity.ready
      yield* TestClock.adjust("1799999 millis")
      expect(Option.isNone(yield* shutdown.current)).toBe(true)
      yield* TestClock.adjust("1 millis")
      expect((yield* shutdown.await).reason).toBe("idle")
    }).pipe(Effect.provide(layer))
    await Effect.runPromise(program)
  })
})
