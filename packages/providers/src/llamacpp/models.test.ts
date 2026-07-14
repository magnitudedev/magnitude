import { describe, expect, it } from "vitest"
import type { ChatCompletionsRequest } from "@magnitudedev/ai"
import { composeLlamaCppReasoningRequest } from "./models"

function request(reasoningEffort: string): Partial<ChatCompletionsRequest> {
  return {
    model: "local.gguf",
    messages: [{ role: "user", content: "hello" }],
    reasoning_effort: reasoningEffort as ChatCompletionsRequest["reasoning_effort"],
  }
}

describe("composeLlamaCppReasoningRequest", () => {
  it("disables template thinking for none", () => {
    expect(composeLlamaCppReasoningRequest(request("none"))).toEqual({
      model: "local.gguf",
      messages: [{ role: "user", content: "hello" }],
      chat_template_kwargs: { enable_thinking: false },
    })
  })

  it("passes enabled efforts through the template kwargs", () => {
    expect(composeLlamaCppReasoningRequest(request("high"))).toEqual({
      model: "local.gguf",
      messages: [{ role: "user", content: "hello" }],
      chat_template_kwargs: {
        enable_thinking: true,
        reasoning_effort: "high",
      },
    })
  })

  it("omits the control for default", () => {
    expect(composeLlamaCppReasoningRequest(request("default"))).toEqual({
      model: "local.gguf",
      messages: [{ role: "user", content: "hello" }],
    })
  })
})
