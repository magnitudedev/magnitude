import * as HttpClient from "@effect/platform/HttpClient"
import {
  type GeneratedClientError,
  type GeneratedClientOptions,
} from "@magnitudedev/openapi-effect/client-runtime"
import { makeIcnApiClient } from "@magnitudedev/icn-protocol/client"
import * as S from "@magnitudedev/icn-protocol/schemas"
import { Effect, Option, Stream } from "effect"
import { MAGNITUDE_INFERENCE_BASE_URL } from "./inference-endpoint"

export const ResponseObjectSchema = S.ResponseObject

export type ResponseObject = typeof ResponseObjectSchema.Type

/** Typed client for the public `/inference/v1` data plane. */
export interface InferenceClient {
  readonly createChatCompletion: (
    payload: typeof S.ChatCompletionRequest.Type,
  ) => Effect.Effect<typeof S.ChatCompletionResponse.Type, GeneratedClientError>
  readonly streamChatCompletion: (
    payload: typeof S.ChatCompletionRequest.Type,
    options?: { readonly includeProgress?: boolean },
  ) => Stream.Stream<typeof S.ChatCompletionChunk.Type, GeneratedClientError>
  readonly streamResponse: (
    payload: typeof S.ResponseCreateRequest.Type,
    options?: { readonly includeProgress?: boolean },
  ) => Stream.Stream<typeof S.ResponseStreamEvent.Type, GeneratedClientError>
  readonly createResponse: (
    payload: typeof S.ResponseCreateRequest.Type,
  ) => Effect.Effect<ResponseObject, GeneratedClientError>
}

export const makeInferenceClient = (
  options: GeneratedClientOptions = { baseUrl: MAGNITUDE_INFERENCE_BASE_URL },
): Effect.Effect<InferenceClient, never, HttpClient.HttpClient> => Effect.gen(function* () {
  const transport = yield* makeIcnApiClient(options)
  const progressHeaders = (includeProgress: boolean | undefined) => ({
    "Magnitude-Include-Progress": Option.fromNullable(includeProgress),
  })
  return {
    createChatCompletion: (payload) => transport.chat.createChatCompletionHttp({
      payload: { ...payload, stream: Option.some(false) },
      headers: progressHeaders(false),
    }),
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
    createResponse: (payload) => transport.inference.createResponseHttp({
      payload: { ...payload, stream: Option.some(false) },
      headers: progressHeaders(false),
    }),
  }
})
