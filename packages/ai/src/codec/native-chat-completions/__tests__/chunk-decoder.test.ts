import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { standardChatCompletionChunkDecoder } from "../chunk-decoder"

const payload = (delta: Record<string, unknown>) => JSON.stringify({
  id: "chatcmpl-test",
  object: "chat.completion.chunk",
  created: 1,
  model: "test-model",
  choices: [{
    index: 0,
    delta,
    finish_reason: "stop",
  }],
  usage: {
    prompt_tokens: 10,
    completion_tokens: 5,
  },
})

describe("standard chat-completions chunk decoder", () => {
  it("normalizes the standard wire payload before stream reduction", async () => {
    const chunk = await Effect.runPromise(
      standardChatCompletionChunkDecoder.decode(payload({
        content: "answer",
        reasoning_content: "thought",
      })),
    )

    expect(Option.getOrThrow(chunk.delta)).toMatchObject({
      text: Option.some("answer"),
      thought: Option.some("thought"),
    })
    expect(chunk.finishReason).toEqual(Option.some("stop"))
    expect(chunk.usage).toEqual(Option.some({
      inputTokens: 10,
      outputTokens: 5,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      cost: null,
    }))
  })
})
