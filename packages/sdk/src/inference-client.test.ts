import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import * as S from "@magnitudedev/icn-protocol/schemas"
import { Effect, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { makeInferenceClient } from "./inference-client"

describe("standalone inference streaming client", () => {
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
