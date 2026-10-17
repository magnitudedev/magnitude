import { describe, expect, it } from "vitest"
import { consumeChatEvent } from "./chat-stream"

describe("consumeChatEvent", () => {
  it("emits incremental text deltas", () => {
    const chunks: string[] = []
    expect(consumeChatEvent('data: {"choices":[{"delta":{"content":"Hello"}}]}', value => chunks.push(value))).toBe(false)
    expect(consumeChatEvent('data: {"choices":[{"delta":{"content":" world"}}]}', value => chunks.push(value))).toBe(false)
    expect(chunks).toEqual(["Hello", " world"])
  })

  it("recognizes the terminal marker", () => {
    expect(consumeChatEvent("data: [DONE]", () => undefined)).toBe(true)
  })

  it("ignores non-data lines and empty data", () => {
    const chunks: string[] = []
    expect(consumeChatEvent(": keepalive\nevent: message\ndata:", value => chunks.push(value))).toBe(false)
    expect(chunks).toEqual([])
  })

  it("surfaces server-side error events", () => {
    expect(() => consumeChatEvent('data: {"error":{"message":"model unavailable"}}', () => undefined)).toThrow("model unavailable")
  })

  it("rejects malformed JSON data rather than silently losing a response", () => {
    expect(() => consumeChatEvent("data: {not-json}", () => undefined)).toThrow()
  })
})
