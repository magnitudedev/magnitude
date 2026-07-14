import { describe, it, expect, beforeEach } from "vitest"
import { Stream, Effect, Chunk } from "effect"
import { decode } from "../decode"
import { acceptedHttpResponse, type StreamFailureContext } from "../../../errors/failure"
import type { ChatCompletionsStreamChunk } from "../../../wire/chat-completions"
import type { ResponseStreamEvent } from "../../../response/events"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function chunkFromData(data: Partial<ChatCompletionsStreamChunk>): ChatCompletionsStreamChunk {
  return {
    id: data.id ?? "chatcmpl-test",
    object: data.object ?? "chat.completion.chunk",
    created: data.created ?? 1234567890,
    model: data.model ?? "test-model",
    choices: data.choices ?? [],
    usage: data.usage,
    error: data.error,
  } as ChatCompletionsStreamChunk
}

function textChunk(content: string): ChatCompletionsStreamChunk {
  return chunkFromData({
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  })
}

function usageChunk(): ChatCompletionsStreamChunk {
  return chunkFromData({
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 10, completion_tokens: 5 },
  })
}

function errorChunk(
  message: string,
  type = "server_error",
  code = "stream_interrupted",
): ChatCompletionsStreamChunk {
  return chunkFromData({
    choices: [],
    error: { message, type, code, param: null },
  })
}

const responseHeaders = new Headers({ "x-request-id": "request-test" })
const streamContext: StreamFailureContext = {
  responseHeaders,
  call: {
    provider: "test-provider",
    model: "test-model",
    method: "POST",
    url: "https://example.test/chat/completions",
  },
  response: acceptedHttpResponse(200, responseHeaders),
}

/** Collect all events from a stream synchronously. */
async function collectEvents(
  events: Stream.Stream<ResponseStreamEvent, never>,
): Promise<ResponseStreamEvent[]> {
  const chunk = await Effect.runPromise(Stream.runCollect(events))
  return Chunk.toArray(chunk)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("decode — mid-stream error envelope", () => {
  it("decodes the reasoning alias as thought events", async () => {
    const chunks = Stream.fromIterable([
      chunkFromData({
        choices: [{ index: 0, delta: { reasoning: "thinking" }, finish_reason: null }],
      }),
      usageChunk(),
    ])
    const { events } = decode(chunks, { streamContext, toStreamFailure: (error) => error })

    const result = await collectEvents(events)

    expect(result.filter((event) => event._tag === "thought_delta")).toEqual([
      { _tag: "thought_delta", text: "thinking" },
    ])
  })

  it("emits provider reasoning details without interpreting them", async () => {
    const details = [{
      type: "reasoning.encrypted",
      data: "opaque-signed-payload",
      format: "provider-v1",
      index: 0,
    }]
    const chunks = Stream.fromIterable([
      chunkFromData({
        choices: [{ index: 0, delta: { reasoning_details: details }, finish_reason: null }],
      }),
      usageChunk(),
    ])
    const { events } = decode(chunks, { streamContext, toStreamFailure: (error) => error })

    const result = await collectEvents(events)

    expect(result.filter((event) => event._tag === "reasoning_details")).toEqual([
      { _tag: "reasoning_details", details },
    ])
  })

  it("emits stream_end with StreamFailed terminal when chunk has error field", async () => {
    const chunks = Stream.fromIterable([
      textChunk("Hello, "),
      textChunk("world!"),
      errorChunk("upstream provider is unavailable", "server_error", "upstream_unavailable"),
    ])

    const { events } = decode(chunks, {
      streamContext,
      toStreamFailure: (e) => e,
    })

    const result = await collectEvents(events)

    // Should see content before the error
    const messageStarts = result.filter((e) => e._tag === "message_start")
    const messageDeltas = result.filter((e) => e._tag === "message_delta")
    expect(messageStarts).toHaveLength(1)
    expect(messageDeltas).toHaveLength(2)
    expect(messageDeltas.map((e) => e.text)).toEqual(["Hello, ", "world!"])

    // The final event must be stream_end with StreamFailed terminal and a concrete cause.
    const streamEnd = result[result.length - 1]
    expect(streamEnd._tag).toBe("stream_end")
    if (streamEnd._tag === "stream_end") {
      expect(streamEnd.terminal._tag).toBe("StreamFailed")
      if (streamEnd.terminal._tag === "StreamFailed") {
        const { cause } = streamEnd.terminal
        expect(cause._tag).toBe("StreamProviderError")
        if (cause._tag !== "StreamProviderError") return
        expect(cause.providerError.code).toBe("upstream_unavailable")
      }
    }
  })

  it("transitions to DONE phase and ignores subsequent chunks", async () => {
    const chunks = Stream.fromIterable([
      textChunk("Hello"),
      errorChunk("stream interrupted", "server_error", "stream_interrupted"),
      textChunk("this should be ignored"),
      textChunk("this too"),
    ])

    const { events } = decode(chunks, {
      streamContext,
      toStreamFailure: (e) => e,
    })

    const result = await collectEvents(events)

    // Only content before error and the stream_end should appear
    const messageDeltas = result.filter((e) => e._tag === "message_delta")
    expect(messageDeltas).toHaveLength(1)
    expect(messageDeltas[0].text).toBe("Hello")

    // No content after error
    const allDeltas = result.filter((e) => e._tag === "message_delta")
    expect(allDeltas).toHaveLength(1)

    // Final event is stream_end
    const streamEnd = result[result.length - 1]
    expect(streamEnd._tag).toBe("stream_end")
    if (streamEnd._tag === "stream_end") {
      expect(streamEnd.terminal._tag).toBe("StreamFailed")
    }
  })

  it("error envelope with missing optional fields still works", async () => {
    // errorChunk uses code="stream_interrupted" — also test minimal error shape
    const chunks = Stream.fromIterable([
      chunkFromData({
        choices: [],
        // Only required field: message; type, code, param all optional
        error: { message: "something went wrong" },
      } as Partial<ChatCompletionsStreamChunk> as ChatCompletionsStreamChunk),
    ])

    const { events } = decode(chunks, {
      streamContext,
      toStreamFailure: (e) => e,
    })

    const result = await collectEvents(events)

    expect(result).toHaveLength(1)
    expect(result[0]._tag).toBe("stream_end")
    if (result[0]._tag === "stream_end") {
      expect(result[0].terminal._tag).toBe("StreamFailed")
      if (result[0].terminal._tag === "StreamFailed") {
        expect(result[0].terminal.cause._tag).toBe("StreamProviderError")
      }
    }
  })

  it("normal stream completion still works when no error envelope is present", async () => {
    const chunks = Stream.fromIterable([
      textChunk("Hello, "),
      textChunk("world!"),
      usageChunk(),
    ])

    const { events } = decode(chunks, {
      streamContext,
      toStreamFailure: (e) => e,
    })

    const result = await collectEvents(events)

    const streamEnd = result[result.length - 1]
    expect(streamEnd._tag).toBe("stream_end")
    if (streamEnd._tag === "stream_end") {
      expect(streamEnd.terminal._tag).toBe("StreamCompleted")
      if (streamEnd.terminal._tag === "StreamCompleted") {
        expect(streamEnd.terminal.usage._tag).toBe("UsageReported")
        if (streamEnd.terminal.usage._tag === "UsageReported") {
          expect(streamEnd.terminal.usage.usage).toEqual({
            inputTokens: 10,
            outputTokens: 5,
            cacheReadTokens: 0,
            cacheWriteTokens: 0,
            cost: null,
          })
        }
      }
    }
  })

  it("error envelope with empty choices array works", async () => {
    const chunks = Stream.fromIterable([
      textChunk("partial response"),
      // Error chunk with choices:[] (same shape as usage chunks)
      errorChunk("server error mid-stream", "server_error", "internal_server_error"),
    ])

    const { events } = decode(chunks, {
      streamContext,
      toStreamFailure: (e) => e,
    })

    const result = await collectEvents(events)

    const streamEnd = result[result.length - 1]
    expect(streamEnd._tag).toBe("stream_end")
    if (streamEnd._tag === "stream_end") {
      expect(streamEnd.terminal._tag).toBe("StreamFailed")
    }
  })

  it("emits StreamOperationalFailure when the stream closes before any chunk", async () => {
    const { events } = decode(Stream.empty, {
      streamContext,
      toStreamFailure: (e) => e,
    })

    const result = await collectEvents(events)

    expect(result).toHaveLength(1)
    expect(result[0]._tag).toBe("stream_end")
    if (result[0]._tag !== "stream_end") return
    expect(result[0].terminal._tag).toBe("StreamFailed")
    if (result[0].terminal._tag !== "StreamFailed") return
    expect(result[0].terminal.cause._tag).toBe("StreamOperationalFailure")
    if (result[0].terminal.cause._tag !== "StreamOperationalFailure") return
    expect(result[0].terminal.cause.reason._tag).toBe("ConnectionClosedWithoutTerminalOutcome")
    if (result[0].terminal.cause.reason._tag !== "ConnectionClosedWithoutTerminalOutcome") return
    expect(result[0].terminal.cause.reason.expectation._tag).toBe("InitialChunk")
  })

  it("emits StreamOperationalFailure when the stream closes before finish_reason", async () => {
    const chunks = Stream.fromIterable([
      textChunk("partial response"),
    ])

    const { events } = decode(chunks, {
      streamContext,
      toStreamFailure: (e) => e,
    })

    const result = await collectEvents(events)

    const streamEnd = result[result.length - 1]
    expect(streamEnd._tag).toBe("stream_end")
    if (streamEnd._tag !== "stream_end") return
    expect(streamEnd.terminal._tag).toBe("StreamFailed")
    if (streamEnd.terminal._tag !== "StreamFailed") return
    expect(streamEnd.terminal.cause._tag).toBe("StreamOperationalFailure")
    if (streamEnd.terminal.cause._tag !== "StreamOperationalFailure") return
    expect(streamEnd.terminal.cause.reason._tag).toBe("ConnectionClosedWithoutTerminalOutcome")
    if (streamEnd.terminal.cause.reason._tag !== "ConnectionClosedWithoutTerminalOutcome") return
    expect(streamEnd.terminal.cause.reason.expectation._tag).toBe("FinishReasonOrMoreChunks")
  })
})
