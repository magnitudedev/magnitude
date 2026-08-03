import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { makeAcnServiceLifecycle } from "./service-lifecycle"

describe("AcnServiceLifecycle", () => {
  it("keeps readiness, RPC availability, admission, and stopping coherent", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const lifecycle = yield* makeAcnServiceLifecycle()
      yield* lifecycle.reportStarting("Resolving", Option.none())
      const starting = yield* lifecycle.state
      yield* lifecycle.becomeReady(Effect.die("unused RPC"))
      const ready = yield* lifecycle.state
      expect(yield* lifecycle.beginStopping({ reason: "peer-request" })).toBe(true)
      expect(yield* lifecycle.beginStopping({ reason: "fatal" })).toBe(false)
      return { starting, ready, stopping: yield* lifecycle.state }
    })))
    expect(result.starting._tag).toBe("Starting")
    expect(result.ready._tag).toBe("Ready")
    expect(result.stopping).toMatchObject({ _tag: "Stopping", reason: "peer-request" })
  })
})
