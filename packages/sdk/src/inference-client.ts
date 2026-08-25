import * as HttpClient from "@effect/platform/HttpClient"
import type { GeneratedClientError, GeneratedClientOptions } from "@magnitudedev/openapi-effect/client-runtime"
import { makeIcnApiClient } from "@magnitudedev/icn-protocol/client"
import * as S from "@magnitudedev/icn-protocol/schemas"
import { Effect, Option, Stream } from "effect"
import { MAGNITUDE_INFERENCE_BASE_URL } from "./inference-endpoint"

/** Streaming-only client for the public `/inference/v1` data plane. */
export interface InferenceClient {
  readonly streamChatCompletion: (
    payload: typeof S.ChatCompletionRequest.Type,
    options?: { readonly includeProgress?: boolean },
  ) => Stream.Stream<typeof S.ChatCompletionChunk.Type, GeneratedClientError>
  readonly streamResponse: (
    payload: typeof S.ResponseCreateRequest.Type,
    options?: { readonly includeProgress?: boolean },
  ) => Stream.Stream<typeof S.ResponseStreamEvent.Type, GeneratedClientError>
}

export const makeInferenceClient = (
  options: GeneratedClientOptions = { baseUrl: MAGNITUDE_INFERENCE_BASE_URL },
): Effect.Effect<InferenceClient, never, HttpClient.HttpClient> => Effect.gen(function* () {
  const transport = yield* makeIcnApiClient(options)
  const progressHeaders = (includeProgress: boolean | undefined) => ({
    "Magnitude-Include-Progress": Option.fromNullable(includeProgress),
  })
  return {
    streamChatCompletion: (payload, streamOptions) => Stream.unwrap(Effect.map(
      transport.chat.createChatCompletion({
        payload,
        headers: progressHeaders(streamOptions?.includeProgress),
      }),
      ({ events }) => events,
    )),
    streamResponse: (payload, streamOptions) => Stream.unwrap(Effect.map(
      transport.inference.createResponse({
        payload,
        headers: progressHeaders(streamOptions?.includeProgress),
      }),
      ({ events }) => events,
    )),
  }
})
