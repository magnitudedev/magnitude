import { describe, expect, it } from "vitest"
import { titleOf } from "./session-record"

describe("chat titles", () => {
  it("keep a short message whole, with its whitespace collapsed", () => {
    expect(titleOf("  Rename the\n\nphotos  ")).toBe("Rename the photos")
  })

  it("cut a long message at a word and mark the cut", () => {
    const title = titleOf("Summarize the attached notes from the Q4 planning meeting and list every attendee with their role please.")
    expect(title).toBe("Summarize the attached notes from the Q4 planning meeting and list every…")
    expect([...title].length).toBeLessThanOrEqual(80)
  })

  it("cut a long word that has no space to break at", () => {
    expect([...titleOf("x".repeat(200))].length).toBe(80)
  })
})
