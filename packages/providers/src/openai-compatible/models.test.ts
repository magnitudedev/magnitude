import { describe, expect, it } from "vitest"
import type { ChatCompletionsRequest } from "@magnitudedev/ai"
import { composeReasoningRequest } from "./models"

function request(reasoningEffort: string): ChatCompletionsRequest {
  return {
    model: "test-model",
    messages: [],
    stream: true,
    reasoning_effort: reasoningEffort as ChatCompletionsRequest["reasoning_effort"],
  }
}

describe("OpenAI-compatible reasoning requests", () => {
  it("uses OpenRouter's normalized reasoning object", () => {
    expect(composeReasoningRequest(request("high"), "openrouter")).toEqual({
      model: "test-model",
      messages: [],
      stream: true,
      reasoning: { effort: "high" },
    })
  })

  it("combines thinking mode and effort for DeepSeek-style APIs", () => {
    expect(composeReasoningRequest(request("max"), "thinking-effort")).toEqual({
      model: "test-model",
      messages: [],
      stream: true,
      reasoning_effort: "max",
      thinking: { type: "enabled" },
    })
    expect(composeReasoningRequest(request("none"), "thinking-effort")).toEqual({
      model: "test-model",
      messages: [],
      stream: true,
      thinking: { type: "disabled" },
    })
  })

  it("uses Kimi's thinking effort and preserves tool-call reasoning", () => {
    expect(composeReasoningRequest(request("high"), "kimi")).toEqual({
      model: "test-model",
      messages: [],
      stream: true,
      thinking: { type: "enabled", effort: "high", keep: "all" },
    })
  })

  it("omits reasoning controls when the provider catalog exposes no control", () => {
    expect(composeReasoningRequest(request("default"), "openai")).toEqual({
      model: "test-model",
      messages: [],
      stream: true,
    })
  })
})
