import { Deferred, Effect, Fiber, Layer, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { AcnActivityTracker, AcnActivityTrackerLive } from "./activity-tracker"
import { AcnServiceLifecycle, AcnServiceLifecycleLive } from "./service-lifecycle"

const TestLayer = AcnActivityTrackerLive.pipe(
  Layer.provideMerge(AcnServiceLifecycleLive("30 minutes")),
  Layer.provideMerge(TestContext.TestContext),
)

describe("AcnActivityTracker", () => {
  it("enters stopping at the exact idle deadline", async () => {
    const program = Effect.gen(function* () {
      const activity = yield* AcnActivityTracker
      const lifecycle = yield* AcnServiceLifecycle
      yield* lifecycle.becomeReady(Effect.die("unused RPC"))
      yield* TestClock.adjust("1799999 millis")
      expect((yield* lifecycle.state)._tag).toBe("Ready")
      yield* TestClock.adjust("1 millis")
      expect((yield* lifecycle.awaitStopping).reason).toBe("idle")
      expect((yield* activity.current).phase).toBe("retired")
    }).pipe(Effect.provide(TestLayer))
    await Effect.runPromise(program)
  })

  it("protects in-flight demand and starts a full interval on release", async () => {
    const program = Effect.gen(function* () {
      const activity = yield* AcnActivityTracker
      const lifecycle = yield* AcnServiceLifecycle
      yield* lifecycle.becomeReady(Effect.die("unused RPC"))
      const latch = yield* Deferred.make<void>()
      const fiber = yield* activity.withUse("work", Deferred.await(latch)).pipe(Effect.fork)
      yield* TestClock.adjust("2 hours")
      expect((yield* lifecycle.state)._tag).toBe("Ready")
      yield* Deferred.succeed(latch, undefined)
      yield* Fiber.join(fiber)
      yield* TestClock.adjust("30 minutes")
      expect((yield* lifecycle.awaitStopping).reason).toBe("idle")
    }).pipe(Effect.provide(TestLayer))
    await Effect.runPromise(program)
  })
})
