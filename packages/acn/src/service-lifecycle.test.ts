import { Duration, Effect, Option, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { makeAcnServiceLifecycle } from "./service-lifecycle"

const run = <A, E>(effect: Effect.Effect<A, E, never>) =>
  Effect.runPromise(Effect.provide(effect, TestContext.TestContext))

describe("AcnServiceLifecycle", () => {
  it("keeps readiness, RPC availability, and stopping coherent", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const lifecycle = yield* makeAcnServiceLifecycle()
      yield* lifecycle.reportStarting("Resolving", Option.none())
      const starting = yield* lifecycle.state
      yield* lifecycle.becomeReady(Effect.die("unused RPC"))
      const ready = yield* lifecycle.state
      expect(yield* lifecycle.beginStopping({ reason: "replacement" })).toBe(true)
      expect(yield* lifecycle.beginStopping({ reason: "fatal" })).toBe(false)
      return { starting, ready, stopping: yield* lifecycle.state }
    })))
    expect(result.starting._tag).toBe("Starting")
    expect(result.ready._tag).toBe("Ready")
    expect(result.stopping).toMatchObject({ _tag: "Stopping", reason: "replacement" })
  })

  it("can stop during startup", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const lifecycle = yield* makeAcnServiceLifecycle()
      expect(yield* lifecycle.beginStopping({ reason: "startup-failed" })).toBe(true)
      expect((yield* lifecycle.awaitStopping)._tag).toBe("Stopping")
    })))
  })

  it("starts the first full idle interval at readiness when no client is present", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const lifecycle = yield* makeAcnServiceLifecycle("30 minutes")
      yield* lifecycle.becomeReady(Effect.die("unused RPC"))
      yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.minutes(30).pipe(Duration.subtract(Duration.millis(1))))
      expect((yield* lifecycle.state)._tag).toBe("Ready")
      yield* TestClock.adjust(Duration.millis(1))
      expect((yield* lifecycle.awaitStopping).reason).toBe("idle")
    })))
  })

  it("fences stale idle timers and starts a fresh interval on final client departure", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const lifecycle = yield* makeAcnServiceLifecycle("30 minutes")
      yield* lifecycle.becomeReady(Effect.die("unused RPC"))
      yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.minutes(29))
      expect(yield* lifecycle.setClientPresence(true)).toBe(true)
      yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.minutes(2))
      expect((yield* lifecycle.state)._tag).toBe("Ready")

      expect(yield* lifecycle.setClientPresence(false)).toBe(true)
      yield* Effect.yieldNow()
      yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.minutes(30).pipe(Duration.subtract(Duration.millis(1))))
      expect((yield* lifecycle.state)._tag).toBe("Ready")
      yield* TestClock.adjust(Duration.millis(1))
      yield* Effect.yieldNow()
      expect(yield* lifecycle.state).toMatchObject({ _tag: "Stopping", reason: "idle" })
      expect(yield* lifecycle.setClientPresence(true)).toBe(false)
    })))
  })
})
