import { describe, expect, it } from "vitest"
import { SessionNotFound, SessionStartFailed } from "@magnitudedev/protocol"
import { formatUnknownCause, toSessionError } from "./session-errors"

describe("session error formatting", () => {
  it("preserves protocol session errors", () => {
    const error = new SessionNotFound({ sessionId: "s1" })
    expect(toSessionError("s1", error)).toBe(error)
  })

  it("keeps tagged addressed-style cause details in SessionStartFailed reasons", () => {
    const result = toSessionError("s1", {
      _tag: "AddressedStoreError",
      operation: "load",
      namespace: "DisplayTimeline/messages",
      address: "DisplayTimeline/messages/forks/root/entries/entry-0",
      cause: new Error("entry file unreadable"),
    })

    expect(result).toBeInstanceOf(SessionStartFailed)
    expect(result._tag).toBe("SessionStartFailed")
    if (result._tag === "SessionStartFailed") {
      expect(result.reason).toContain("AddressedStoreError")
      expect(result.reason).toContain("DisplayTimeline/messages")
      expect(result.reason).toContain("entry file unreadable")
    }
  })

  it("formats plain errors by message", () => {
    expect(formatUnknownCause(new Error("plain failure"))).toBe("plain failure")
  })
})
