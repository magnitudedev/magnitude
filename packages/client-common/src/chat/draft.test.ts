import { describe, expect, it } from "vitest"
import { Effect, Option, Schema } from "effect"
import { ChatAssistantMessage, ChatMessage, MessageId, ServingModelId, SessionId } from "@magnitudedev/sdk"
import {
  type ChatDraft,
  chatComposerText,
  chatRequest,
  chatRequestSettings,
  chatRetry,
  chatSend,
  completeChatSubmission,
  defaultResponseSettings,
} from "./draft"

const id = MessageId.make("response")
const draft: ChatDraft = { text: "Submitted text", settings: {}, submitted: null }
const failed = Schema.decodeUnknownSync(ChatAssistantMessage)({
  type: "assistant",
  id,
  model: "qwen3-8b",
  settings: { systemPrompt: "", stop: [] },
  started: 0,
  content: [],
  end: { time: 1, outcome: { type: "failed", error: { type: "unreachable", message: "Closed" } } },
})
const completed = Schema.decodeUnknownSync(ChatAssistantMessage)({
  type: "assistant",
  id,
  model: "qwen3-8b",
  settings: { systemPrompt: "", stop: [] },
  started: 0,
  content: [],
  end: { time: 1, outcome: { type: "finished", finishReason: "stop" } },
})
const request = { model: ServingModelId.make("qwen3-8b"), reasoningEffort: Option.none(), settings: defaultResponseSettings }
const chatId = SessionId.make("chat")

describe("Chat draft lifecycle", () => {
  it("clears only the submitted text and restores it after a failed response", () => {
    const next = completeChatSubmission(draft, draft.text, id, false)
    expect(next.text).toBe("")
    expect(chatComposerText(next, failed)).toBe(draft.text)
    expect(chatComposerText(next, completed)).toBe("")
  })
  it("preserves text typed while the send is pending", () => {
    const next = completeChatSubmission({ ...draft, text: "Next message" }, draft.text, id, false)
    expect(next.text).toBe("Next message")
    expect(chatComposerText(next, failed)).toBe("Next message")
  })
  it("does not erase the next draft when retrying a response", () => {
    expect(completeChatSubmission(draft, "Previous question", id, true).text).toBe(draft.text)
  })
  it("does not restore an unrelated failed response", () => {
    expect(
      chatComposerText({ ...draft, text: "", submitted: { id: MessageId.make("other"), text: "Other" } }, failed)
    ).toBe("")
  })
})

describe("Chat requests", () => {
  it("gives every send and retry new IDs", () => {
    const first = Effect.runSync(chatSend(chatId, "Hi", request))
    const second = Effect.runSync(chatSend(chatId, "Hi", request))
    expect(new Set([first.messageId, first.responseId, second.messageId, second.responseId]).size).toBe(4)
    const retry = Effect.runSync(chatRetry(chatId, first.responseId, request))
    expect(retry.replaces).toBe(first.responseId)
    expect(retry.responseId).not.toBe(first.responseId)
  })

  it("starts from the transcript's latest response's settings, else the defaults", () => {
    expect(chatRequestSettings([])).toEqual({ ...defaultResponseSettings, model: Option.none(), reasoningEffort: Option.none() })
    const messages = Schema.decodeUnknownSync(Schema.Array(ChatMessage))([
      { type: "user", id: "message-1", text: "Hi", time: 0 },
      { type: "assistant", id, model: "qwen3-8b", reasoningEffort: "high", settings: { systemPrompt: "Be brief.", temperature: 0.5, stop: ["END"] }, started: 0, content: [] },
      { type: "user", id: "message-2", text: "Again", time: 1 },
    ])
    const settings = chatRequestSettings(messages)
    expect(settings.model).toEqual(Option.some("qwen3-8b"))
    expect(settings.reasoningEffort).toEqual(Option.some("high"))
    expect(settings.systemPrompt).toBe("Be brief.")
    expect(settings.temperature).toEqual(Option.some(0.5))
    expect(settings.stop).toEqual(["END"])
  })

  it("makes a request only once a model is chosen", () => {
    const settings = chatRequestSettings([])
    expect(chatRequest(settings)).toEqual(Option.none())
    expect(chatRequest({ ...settings, model: Option.some(request.model) })).toEqual(Option.some(request))
  })
})
