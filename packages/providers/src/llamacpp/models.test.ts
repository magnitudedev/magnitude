import { describe, expect, it } from "vitest"
import { Effect, Layer, Stream } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import type * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { Auth, Prompt, type ToolDefinition } from "@magnitudedev/ai"
import { createLlamaCppCompatibleSpec } from "./models"

const prompt = Prompt.from({
  system: "You are helpful.",
  messages: [{ _tag: "UserMessage", parts: [{ _tag: "TextPart", text: "hello" }] }],
})
const tools: readonly ToolDefinition[] = []

const requestBodyText = (request: HttpClientRequest.HttpClientRequest): string => {
  const body = request.body
  if (body._tag === "Uint8Array") return new TextDecoder().decode(body.body)
  if (body._tag === "Raw" && typeof body.body === "string") return body.body
  throw new Error(`Unexpected request body: ${body._tag}`)
}

describe("llama.cpp request options", () => {
  it("sends template kwargs and thinking budget through separate request fields", async () => {
    let body: unknown
    const mockClient = HttpClient.make((request) => {
      body = JSON.parse(requestBodyText(request))
      return Effect.succeed(HttpClientResponse.fromWeb(
        request,
        new Response([
          'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"test","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}',
          "",
          "data: [DONE]",
          "",
        ].join("\n"), { status: 200, headers: { "content-type": "text/event-stream" } }),
      ))
    })
    const model = createLlamaCppCompatibleSpec({
      modelId: "test-model",
      endpoint: "http://127.0.0.1:8080/v1",
    }).bind({
      auth: Auth.bearer("test"),
      defaults: {
        chatTemplateKwargs: { enable_thinking: true, reasoning_effort: "high" },
        thinkingBudgetTokens: 4_096,
      },
    })

    await Effect.runPromise(Effect.gen(function* () {
      const response = yield* model.stream(prompt, tools)
      yield* Stream.runDrain(response.events)
    }).pipe(Effect.provide(Layer.succeed(HttpClient.HttpClient, mockClient))))

    expect(body).toMatchObject({
      chat_template_kwargs: { enable_thinking: true, reasoning_effort: "high" },
      thinking_budget_tokens: 4_096,
    })
  })
})
