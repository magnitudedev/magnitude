import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import {
  ChatConversation,
  ChatGeneration,
  ChatSettings,
  applyChatUpdate,
  defaultChatSettings,
} from "./chat"
const settings = Schema.encodeSync(ChatSettings)(defaultChatSettings)
const g = Schema.decodeUnknownSync(ChatGeneration)({
  id: "gen",
  commandId: "cmd",
  messageId: "assistant",
  modelId: "model",
  settings,
  status: "running",
  createdAt: 0,
  updatedAt: 100,
  content: "New output",
  reasoning: "",
  metrics: {},
})
const conversation = Schema.decodeUnknownSync(ChatConversation)({
  id: "chat",
  title: "Question",
  createdAt: 0,
  updatedAt: 0,
  settings,
  messages: [
    { id: "user", role: "user", content: "Question" },
    { id: "assistant", role: "assistant", content: "" },
  ],
})
describe("Chat wire contracts", () => {
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
    ])
      expect(
        Schema.decodeUnknownEither(ChatSettings)({ ...settings, ...patch })._tag
      ).toBe("Left")
  })
  it("serializes absent options as omitted fields and preserves explicit zero settings", () => {
    expect(settings).not.toHaveProperty("modelId")
    const decoded = Schema.decodeUnknownSync(ChatSettings)({
      ...settings,
      temperature: 0,
      presencePenalty: 0,
    })
    expect(decoded.temperature).toEqual(Option.some(0))
    expect(Schema.encodeSync(ChatSettings)(decoded).temperature).toBe(0)
  })
  it("folds cumulative generation updates without changing earlier messages or current settings", () => {
    const initial = applyChatUpdate(null, { _tag: "Snapshot", conversation })!
    const next = applyChatUpdate(initial, {
      _tag: "Generation",
      generation: g,
    })!
    expect(next.messages[0]).toBe(initial.messages[0])
    expect(next.messages[1]?.content).toBe("New output")
    expect(next.updatedAt).toBe(100)
    expect(next.settings).toBe(initial.settings)
    expect(initial.messages[1]?.content).toBe("")
    expect(
      applyChatUpdate(next, {
        _tag: "Deleted",
        conversationId: conversation.id,
      })
    ).toBeNull()
    expect(
      applyChatUpdate(null, { _tag: "Generation", generation: g })
    ).toBeNull()
  })
  it("keeps the internally stored request out of the public generation encoding", () => {
    const encoded = Schema.encodeSync(ChatGeneration)({
      ...g,
      requestJson: "private internal request",
    } as ChatGeneration)
    expect(encoded).not.toHaveProperty("requestJson")
  })
})
