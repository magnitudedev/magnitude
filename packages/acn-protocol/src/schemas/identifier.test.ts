import { Effect, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { generateMessageId, generateSessionId, MessageId, SessionId } from "./identifier"

describe("identifiers", () => {
  it("generates the repo's 12-character IDs", () => {
    const session = Effect.runSync(generateSessionId)
    const message = Effect.runSync(generateMessageId)
    expect(session).toMatch(/^[a-z][0-9a-z]{11}$/)
    expect(message).toMatch(/^[a-z][0-9a-z]{11}$/)
    expect(Schema.is(SessionId)(session)).toBe(true)
    expect(Schema.is(MessageId)(message)).toBe(true)
  })

  it("accepts any non-empty ID, so stored IDs outlive a generator change", () => {
    expect(Schema.is(MessageId)("short")).toBe(true)
    expect(Schema.is(MessageId)("")).toBe(false)
  })

  it("does not repeat", () => {
    const ids = new Set(Array.from({ length: 1_000 }, () => Effect.runSync(generateMessageId)))
    expect(ids.size).toBe(1_000)
  })
})
