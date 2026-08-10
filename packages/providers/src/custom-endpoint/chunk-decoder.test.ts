import { Chunk, Effect, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  acceptedHttpResponse,
  nativeChatCompletionsCodec,
  type ResponseStreamEvent,
  type StreamFailureContext,
} from "@magnitudedev/ai"
import { customEndpointChunkDecoder } from "./chunk-decoder"

const payload = (delta: Record<string, unknown>) => JSON.stringify({
  id: "chatcmpl-test",
  object: "chat.completion.chunk",
  created: 1,
  model: "z-ai/glm-5.2",
  choices: [{
    index: 0,
    delta,
    finish_reason: null,
  }],
})

const thought = async (delta: Record<string, unknown>) => {
  const chunk = await Effect.runPromise(customEndpointChunkDecoder.decode(payload(delta)))
  return Option.getOrThrow(Option.getOrThrow(chunk.delta).thought)
}

describe("custom endpoint chunk decoder", () => {
  it("normalizes OpenRouter reasoning without duplicating reasoning_details", async () => {
    expect(await thought({
      reasoning: "one thought",
      reasoning_details: [{
        type: "reasoning.text",
        text: "one thought",
      }],
    })).toBe("one thought")
  })

  it("uses textual OpenRouter reasoning details when reasoning is absent", async () => {
    expect(await thought({
      reasoning_details: [
        { type: "reasoning.text", text: "first " },
        { type: "reasoning.encrypted", data: "opaque" },
        { type: "reasoning.text", text: "second" },
      ],
    })).toBe("first second")
  })

  it.each([
    ["reasoning_content", { reasoning_content: "standard thought" }, "standard thought"],
    ["thinking", { thinking: "thinking text" }, "thinking text"],
  ])("normalizes %s", async (_name, delta, expected) => {
    expect(await thought(delta)).toBe(expected)
  })

  it("preserves standard content and tool-call deltas", async () => {
    const chunk = await Effect.runPromise(customEndpointChunkDecoder.decode(payload({
      content: "answer",
      tool_calls: [{
        index: 0,
        id: "call-1",
        type: "function",
        function: { name: "lookup", arguments: "{\"query\":\"x\"}" },
      }],
    })))
    const delta = Option.getOrThrow(chunk.delta)

    expect(delta.text).toEqual(Option.some("answer"))
    expect(delta.toolCalls).toEqual([{
      index: 0,
      providerToolCallId: Option.some("call-1"),
      name: Option.some("lookup"),
      input: Option.some("{\"query\":\"x\"}"),
    }])
  })

  it("keeps OpenRouter reasoning contiguous when chunks contain empty content", async () => {
    const rawChunks = [
      payload({ content: "", reasoning: "first " }),
      payload({ content: "", reasoning: "second" }),
      JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1,
        model: "z-ai/glm-5.2",
        choices: [{
          index: 0,
          delta: { content: "answer" },
          finish_reason: "stop",
        }],
        usage: { prompt_tokens: 1, completion_tokens: 3 },
      }),
    ]
    const chunks = await Effect.runPromise(
      Effect.forEach(rawChunks, customEndpointChunkDecoder.decode),
    )
    const headers = new Headers()
    const streamContext: StreamFailureContext = {
      responseHeaders: headers,
      call: {
        provider: "custom:openrouter",
        model: "z-ai/glm-5.2",
        method: "POST",
        url: "https://openrouter.ai/api/v1/chat/completions",
      },
      response: acceptedHttpResponse(200, headers),
    }
    const decoded = nativeChatCompletionsCodec.decode(Stream.fromIterable(chunks), {
      streamContext,
      toStreamFailure: (error) => error,
    })
    const events = Chunk.toArray(
      await Effect.runPromise(Stream.runCollect(decoded.events)),
    ) as readonly ResponseStreamEvent[]

    expect(events.map((event) => event._tag)).toEqual([
      "thought_start",
      "thought_delta",
      "thought_delta",
      "thought_end",
      "message_start",
      "message_delta",
      "message_end",
      "stream_end",
    ])
  })
})
