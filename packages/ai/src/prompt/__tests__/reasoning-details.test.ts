import { describe, expect, it } from "vitest"
import { mergeReasoningDetails } from "../messages"

describe("mergeReasoningDetails", () => {
  it("coalesces streaming fragments by type and index", () => {
    const first = mergeReasoningDetails([], [
      { type: "reasoning.text", text: "hello", format: "unknown", index: 0 },
    ])
    expect(mergeReasoningDetails(first, [
      { type: "reasoning.text", text: " world", signature: "signed", index: 0 },
      { type: "reasoning.encrypted", data: "opaque", index: 1 },
    ])).toEqual([
      {
        type: "reasoning.text",
        text: "hello world",
        format: "unknown",
        signature: "signed",
        index: 0,
      },
      { type: "reasoning.encrypted", data: "opaque", index: 1 },
    ])
  })

  it("preserves unindexed details in arrival order", () => {
    expect(mergeReasoningDetails([], [{ type: "custom", data: "one" }, "opaque"])).toEqual([
      { type: "custom", data: "one" },
      "opaque",
    ])
  })
})
