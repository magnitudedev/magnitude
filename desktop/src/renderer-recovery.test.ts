import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { makeRendererRecovery } from "./renderer-recovery"

describe("renderer recovery", () => {
  it("bounds automatic reloads and permits explicit recovery after exhaustion", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const recovery = yield* makeRendererRecovery
      expect(yield* recovery.open).toBe(false)
      expect(yield* recovery.crashed).toBe(true)
      expect(yield* recovery.crashed).toBe(true)
      expect(yield* recovery.crashed).toBe(true)
      expect(yield* recovery.crashed).toBe(false)
      expect(yield* recovery.crashed).toBe(false)
      expect(yield* recovery.open).toBe(true)
      expect(yield* recovery.crashed).toBe(true)
    }))
  })
  it("retains a failed initial load for explicit retry without an automatic loop", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const recovery = yield* makeRendererRecovery
      yield* recovery.loadFailed
      expect(yield* recovery.crashed).toBe(false)
      expect(yield* recovery.open).toBe(true)
      expect(yield* recovery.open).toBe(false)
    }))
  })
})
