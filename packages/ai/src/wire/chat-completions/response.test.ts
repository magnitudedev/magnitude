import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ChatCompletionsStreamChunk } from "./response"

const baseChunk = {
  id: "chunk",
  object: "chat.completion.chunk",
  created: 0,
  model: "test-model",
  choices: [],
}

describe("ChatCompletionsStreamChunk", () => {
  it.each([
    ["missing", baseChunk],
    ["null", { ...baseChunk, usage: null }],
  ])("normalizes %s nullable optional fields to Option.none", (_name, input) => {
    const chunk = Schema.decodeUnknownSync(ChatCompletionsStreamChunk)(input)

    expect(Option.isNone(chunk.usage)).toBe(true)
    expect(Option.isNone(chunk.error)).toBe(true)
  })

  it("normalizes nested nullable token details to Option", () => {
    const chunk = Schema.decodeUnknownSync(ChatCompletionsStreamChunk)({
      ...baseChunk,
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        prompt_tokens_details: {
          cached_tokens: null,
        },
      },
    })

    const usage = Option.getOrThrow(chunk.usage)
    const details = Option.getOrThrow(usage.prompt_tokens_details)
    expect(Option.isNone(details.cached_tokens)).toBe(true)
  })

  it("normalizes nullable choice and delta fields to Option.none", () => {
    const chunk = Schema.decodeUnknownSync(ChatCompletionsStreamChunk)({
      ...baseChunk,
      choices: [{
        index: 0,
        delta: {
          role: null,
          content: null,
          reasoning_content: null,
          tool_calls: null,
        },
        finish_reason: null,
        logprobs: null,
      }],
    })

    const choice = chunk.choices[0]!
    expect(Option.isNone(choice.delta.role)).toBe(true)
    expect(Option.isNone(choice.delta.content)).toBe(true)
    expect(Option.isNone(choice.delta.reasoning_content)).toBe(true)
    expect(Option.isNone(choice.delta.tool_calls)).toBe(true)
    expect(Option.isNone(choice.finish_reason)).toBe(true)
    expect(Option.isNone(choice.logprobs)).toBe(true)
  })

  it("normalizes nullable tool-call and logprob fields to Option.none", () => {
    const chunk = Schema.decodeUnknownSync(ChatCompletionsStreamChunk)({
      ...baseChunk,
      choices: [{
        index: 0,
        delta: {
          tool_calls: [
            { index: 0, id: null, type: null, function: null },
            { index: 1, function: { name: null, arguments: null } },
          ],
        },
        logprobs: { content: null },
      }],
    })

    const choice = chunk.choices[0]!
    const toolCalls = Option.getOrThrow(choice.delta.tool_calls)
    expect(Option.isNone(toolCalls[0]!.id)).toBe(true)
    expect(Option.isNone(toolCalls[0]!.type)).toBe(true)
    expect(Option.isNone(toolCalls[0]!.function)).toBe(true)
    const functionDelta = Option.getOrThrow(toolCalls[1]!.function)
    expect(Option.isNone(functionDelta.name)).toBe(true)
    expect(Option.isNone(functionDelta.arguments)).toBe(true)
    const logprobs = Option.getOrThrow(choice.logprobs)
    expect(Option.isNone(logprobs.content)).toBe(true)
  })

  it("uses Option.none for missing non-null extensions", () => {
    const chunk = Schema.decodeUnknownSync(ChatCompletionsStreamChunk)({
      ...baseChunk,
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    })

    const usage = Option.getOrThrow(chunk.usage)
    expect(Option.isNone(usage.cost)).toBe(true)
    expect(Option.isNone(chunk.raw_input)).toBe(true)
    expect(Option.isNone(chunk.raw_output)).toBe(true)
  })

  it.each([
    { ...baseChunk, usage: { prompt_tokens: 10, completion_tokens: 5, cost: null } },
    { ...baseChunk, raw_input: null },
    { ...baseChunk, raw_output: null },
  ])("rejects null for non-null extensions %#", (input) => {
    expect(() => Schema.decodeUnknownSync(ChatCompletionsStreamChunk)(input)).toThrow()
  })
})
