import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import * as S from "@magnitudedev/icn-protocol/schemas"
import { Effect, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { Inference, inferenceClientErrorMessage } from "./inference"
import { makeInferenceClient } from "./inference-client"

describe("standalone inference Effect Query client", () => {
  it("preserves the typed ICN error message for consumers", () => {
    expect(inferenceClientErrorMessage({
      _tag: "GeneratedClientRemoteError",
      operationId: "ensureModelInstance",
      status: 409,
      headers: {},
      body: { error: { code: "model_unavailable", message: "Model is unavailable", retryable: false, type: "model_error" } },
    })).toBe("Model is unavailable")
  })

  it("runs authored queries and mutation synchronization over the generated transport", async () => {
    let generation = 1
    const requests: string[] = []
    const http = HttpClient.make((request) => {
      requests.push(`${request.method} ${new URL(request.url).pathname}`)
      if (request.method === "PUT") {
        generation = 2
        return Effect.succeed(HttpClientResponse.fromWeb(request, new Response(null, { status: 204 })))
      }
      return Effect.succeed(HttpClientResponse.fromWeb(request, Response.json({
        generation,
        idleTimeoutSeconds: 60,
      })))
    })
    const client = await Effect.runPromise(makeInferenceClient({
      baseUrl: "http://127.0.0.1:10100/inference/",
    }).pipe(Effect.provideService(HttpClient.HttpClient, http)))

    expect(await Effect.runPromise(client.query(
      Inference.GetInferenceResidencyPolicy,
      {},
    ))).toEqual({ generation: 1, idleTimeoutSeconds: 60 })
    await Effect.runPromise(client.mutate(
      Inference.SetInferenceResidencyPolicy,
      { generation: 2, idleTimeoutSeconds: 60 },
    ))
    expect(requests).toEqual([
      "GET /inference/api/v1/residency-policy",
      "PUT /inference/api/v1/residency-policy",
      "GET /inference/api/v1/residency-policy",
    ])
    await Effect.runPromise(client.close)
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
    await Effect.runPromise(client.close)
  })
})
