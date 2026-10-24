import { describe, expect, it } from "vitest"
import {
  ChatGenerationId,
  ChatGeneration,
  ChatSettings,
  defaultChatSettings,
} from "@magnitudedev/sdk"
import { Schema } from "effect"
import {
  chatComposerText,
  completeChatSubmission,
  type ChatDraft,
} from "./draft"
const id = ChatGenerationId.make("gen")
const draft: ChatDraft = {
  text: "Submitted text",
  settings: {},
  submitted: null,
}
const failed = Schema.decodeUnknownSync(ChatGeneration)({
  id,
  commandId: "cmd",
  messageId: "msg",
  modelId: "model",
  settings: Schema.encodeSync(ChatSettings)(defaultChatSettings),
  status: "failed",
  createdAt: 0,
  updatedAt: 0,
  content: "",
  reasoning: "",
  metrics: {},
})
describe("Chat draft lifecycle", () => {
  it("clears only the submitted text and restores it after a failed generation", () => {
    const next = completeChatSubmission(draft, draft.text, id, false)
    expect(next.text).toBe("")
    expect(chatComposerText(next, failed)).toBe(draft.text)
    expect(chatComposerText(next, { ...failed, status: "completed" })).toBe("")
  })
  it("preserves text typed while admission is pending", () => {
    const next = completeChatSubmission(
      { ...draft, text: "Next message" },
      draft.text,
      id,
      false
    )
    expect(next.text).toBe("Next message")
    expect(chatComposerText(next, failed)).toBe("Next message")
  })
  it("does not erase the next draft when retrying a response", () => {
    expect(
      completeChatSubmission(draft, "Previous question", id, true).text
    ).toBe(draft.text)
  })
  it("does not restore an unrelated failed turn", () => {
    expect(
      chatComposerText(
        {
          ...draft,
          text: "",
          submitted: { id: ChatGenerationId.make("different"), text: "Other" },
        },
        failed
      )
    ).toBe("")
  })
})
