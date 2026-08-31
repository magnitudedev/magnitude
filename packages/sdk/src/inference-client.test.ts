import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import * as S from "@magnitudedev/icn-protocol/schemas"
import { Effect, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { makeInferenceClient } from "./inference-client"

describe("standalone inference streaming client", () => {
  it("lists enriched inference models through the generated transport", async () => {
    let requestedUrl = ""
    const http = HttpClient.make((request) => {
      requestedUrl = request.url
      return Effect.succeed(HttpClientResponse.fromWeb(request, Response.json({
        object: "list",
        data: [{
          id: "local/model",
          object: "model",
          created: 0,
          owned_by: "magnitude",
          name: "Local Model",
          description: "Local fixture.",
          context_length: 65_536,
          architecture: { input_modalities: ["text"], output_modalities: ["text"] },
          supported_parameters: ["max_tokens", "reasoning"],
          reasoning: {
            supported_efforts: ["none", "high"],
            default_effort: "high",
            default_enabled: true,
            mandatory: false,
          },
          top_provider: { context_length: 65_536, max_completion_tokens: 32_768 },
        }],
      })))
    })
    const client = await Effect.runPromise(makeInferenceClient().pipe(
      Effect.provideService(HttpClient.HttpClient, http),
    ))

    const response = await Effect.runPromise(client.listModels())

    expect(requestedUrl).toBe("http://127.0.0.1:10100/inference/v1/models")
    expect(response.data[0]?.top_provider.max_completion_tokens).toBe(32_768)
  })

  it("maps the friendly progress option to the Magnitude streaming header", async () => {
    let progressHeader: string | undefined
    const http = HttpClient.make((request) => {
      progressHeader = request.headers["magnitude-include-progress"]
      return Effect.succeed(HttpClientResponse.fromWeb(request, new Response("data: [DONE]\n\n", {
        headers: { "content-type": "text/event-stream" },
      })))
    })
    const client = await Effect.runPromise(makeInferenceClient().pipe(
      Effect.provideService(HttpClient.HttpClient, http),
    ))

    const request = Schema.decodeUnknownSync(S.ChatCompletionRequest)({
      model: "gemma-4-26b-a4b-it-qat:gguf:q4",
      messages: [],
      stream: true,
    })
    await Effect.runPromise(Stream.runDrain(client.streamChatCompletion(
      request,
      { includeProgress: true },
    )))

    expect(progressHeader).toBe("true")
  })
})
