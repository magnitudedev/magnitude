import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { RequestedResponseSettings } from "./chat"

const settings = { systemPrompt: "Be brief.", stop: [] }

describe("Requested response settings", () => {
  it("rejects invalid and non-finite sampling values", () => {
    for (const patch of [
      { temperature: -1 },
      { temperature: NaN },
      { temperature: Infinity },
      { maxOutputTokens: 0 },
      { maxOutputTokens: 1.5 },
      { topP: 0 },
      { minP: 2 },
      { seed: -1 },
      { stop: [""] },
      { stop: Array(17).fill("stop") },
    ]) {
      expect(Schema.decodeUnknownEither(RequestedResponseSettings)({ ...settings, ...patch })._tag).toBe("Left")
    }
  })

  it("serializes absent options as omitted fields and preserves explicit zero settings", () => {
    const decoded = Schema.decodeUnknownSync(RequestedResponseSettings)({ ...settings, temperature: 0, presencePenalty: 0 })
    expect(decoded.temperature).toEqual(Option.some(0))
    const encoded = Schema.encodeSync(RequestedResponseSettings)(decoded)
    expect(encoded.temperature).toBe(0)
    expect(encoded).not.toHaveProperty("topP")
  })
})
