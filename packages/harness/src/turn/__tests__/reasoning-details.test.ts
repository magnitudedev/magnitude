import { describe, expect, it } from "vitest"
import { CanonicalAccumulatorReducer } from "../reducers"

describe("canonical reasoning details", () => {
  it("coalesces streamed detail fragments into the assistant message", () => {
    const first = CanonicalAccumulatorReducer.step(CanonicalAccumulatorReducer.initial, {
      _tag: "ReasoningDetails",
      details: [{ type: "reasoning.text", text: "one", index: 0 }],
    })
    const second = CanonicalAccumulatorReducer.step(first, {
      _tag: "ReasoningDetails",
      details: [{ type: "reasoning.text", text: " two", signature: "signed", index: 0 }],
    })

    expect(second.assistantMessage.reasoningDetails).toEqual([
      { type: "reasoning.text", text: "one two", signature: "signed", index: 0 },
    ])
  })
})
