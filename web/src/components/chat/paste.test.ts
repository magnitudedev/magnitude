import { describe, expect, it } from "vitest"
import { MESSAGE_CHARS, PASTE_TOO_LONG, noticeAfterEdit, pasteTooLong } from "./paste"

describe("pasting into a message", () => {
  it("refuses a paste that would overrun the message, such as 2,347,627 characters, rather than cutting it", () => {
    expect(pasteTooLong("", 0, 0, "x".repeat(2_347_627))).toBe(true)
    expect(PASTE_TOO_LONG).toContain("1,048,576")
  })

  it("counts the text already there, less what the paste replaces", () => {
    const text = "y".repeat(1000)
    expect(pasteTooLong(text, 1000, 1000, "x".repeat(MESSAGE_CHARS - 999))).toBe(true)
    expect(pasteTooLong(text, 0, 1000, "x".repeat(MESSAGE_CHARS))).toBe(false)
    expect(pasteTooLong(text, 1000, 1000, "x".repeat(MESSAGE_CHARS - 1000))).toBe(false)
  })
})

describe("the refused-paste notice", () => {
  it("clears once the message is edited or a paste that fits lands, and leaves other notices alone", () => {
    expect(noticeAfterEdit(PASTE_TOO_LONG)).toBeNull()
    expect(noticeAfterEdit("Couldn’t copy to the clipboard.")).toBe("Couldn’t copy to the clipboard.")
    expect(noticeAfterEdit(null)).toBeNull()
  })
})
