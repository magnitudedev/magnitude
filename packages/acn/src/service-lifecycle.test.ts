import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { makeAcnServiceLifecycle } from "./service-lifecycle"

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

})
