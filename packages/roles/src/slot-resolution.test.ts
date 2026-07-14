import { describe, expect, it } from "vitest"
import { resolveReasoningEffort } from "./slot-resolution"

describe("resolveReasoningEffort", () => {
  it("preserves an explicitly supported effort", () => {
    expect(resolveReasoningEffort(
      { reasoningEfforts: ["none", "low", "medium"] },
      "low",
      "high",
    )).toBe("low")
  })

  it("uses the closest supported standard effort below the slot default", () => {
    expect(resolveReasoningEffort(
      { reasoningEfforts: ["none", "medium"] },
      undefined,
      "high",
    )).toBe("medium")
  })

  it("uses none for a model that does not support reasoning effort", () => {
    expect(resolveReasoningEffort(
      { reasoningEfforts: ["none"] },
      undefined,
      "high",
    )).toBe("none")
  })

  it("enables toggle-only reasoning when high is closer than none", () => {
    expect(resolveReasoningEffort(
      { reasoningEfforts: ["none", "high"] },
      undefined,
      "medium",
    )).toBe("high")
  })
})
