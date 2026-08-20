import { describe, expect, it } from "vitest"
import {
  decodeConversationPreferences,
  defaultConversationPreferences,
  encodeConversationPreferences,
} from "./conversation-preferences"

describe("conversation preferences storage", () => {
  it("round-trips the show-thinking preference", () => {
    expect(decodeConversationPreferences(
      encodeConversationPreferences({ showThinking: true }),
    )).toEqual({ showThinking: true })
  })

  it("defaults safely when storage is absent or invalid", () => {
    expect(decodeConversationPreferences(null)).toEqual(defaultConversationPreferences)
    expect(decodeConversationPreferences("not-json")).toEqual(defaultConversationPreferences)
    expect(decodeConversationPreferences('{"showThinking":"yes"}'))
      .toEqual(defaultConversationPreferences)
  })
})
