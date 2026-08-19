import { describe, expect, it } from "vitest"
import { hasUserMessageContent } from "./session-types"

describe("hasUserMessageContent", () => {
  it("accepts text, upload-only, and mention-only messages", () => {
    expect(hasUserMessageContent({ content: "hello", uploads: [], mentions: [] })).toBe(true)
    expect(hasUserMessageContent({
      content: "",
      uploads: [{ type: "raw_text_file", filename: "notes.md", data: "bm90ZXM=" }],
      mentions: [],
    })).toBe(true)
    expect(hasUserMessageContent({
      content: "  ",
      uploads: [],
      mentions: [{
        occurrenceId: "mention-1",
        attachment: { type: "mention_file", path: "README.md" },
        placement: { _tag: "trailing" },
      }],
    })).toBe(true)
  })

  it("rejects messages with no semantic content", () => {
    expect(hasUserMessageContent({ content: " \n ", uploads: [], mentions: [] })).toBe(false)
  })
})
