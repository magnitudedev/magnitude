import { describe, expect, it } from "vitest"
import { Cause, Effect, Option } from "effect"
import { catalogRemovalOutcome } from "./local-model-removals"

describe("catalog removal outcomes", () => {
  it("accepts an actual removal", async () => {
    await expect(Effect.runPromise(catalogRemovalOutcome({
      _tag: "Removed",
      reclaimedBytes: 42,
    }))).resolves.toEqual({})
  })

  it.each([
    ["ExternalOwnership" as const, "model_removal_retained_external"],
    ["SharedMaterial" as const, "model_removal_retained_shared"],
  ])("does not report a retained installation as removed", async (reason, code) => {
    const exit = await Effect.runPromiseExit(catalogRemovalOutcome({ _tag: "Retained", reason }))
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      const failure = Option.getOrThrow(Cause.failureOption(exit.cause))
      expect(failure.code).toBe(code)
    }
  })
})
