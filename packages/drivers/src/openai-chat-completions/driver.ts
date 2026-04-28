import { Effect, Schema, Stream, pipe } from 'effect'
import { HttpClient, HttpClientRequest, HttpClientResponse } from '@effect/platform'
import { sseChunks } from './sse'
import { ChatCompletionsRequest, ChatCompletionsStreamChunk } from '../wire/chat-completions'
import { DriverError } from '../errors'
import type { Driver, DriverCallOptions } from '../driver'

/**
 * OpenAIChatCompletionsDriver
 *
 * Implements the Driver interface for OpenAI-compatible chat completions
 * endpoints that stream via SSE (Server-Sent Events).
 *
 * Compatible with: Fireworks, OpenRouter, Magnitude, LM Studio, Ollama,
 * and any other OpenAI-compatible provider.
 *
 * Requirements (R channel): HttpClient.HttpClient
 * Provided at the app root via FetchHttpClient.layer (Bun ships native fetch).
 */
export const OpenAIChatCompletionsDriver: Driver<
  ChatCompletionsRequest,
  ChatCompletionsStreamChunk
> = {
  id: 'openai-chat-completions',

  send: (request, options) =>
    Effect.gen(function* () {
      const httpClient = yield* HttpClient.HttpClient

      // Build the HTTP request
      const baseReq = HttpClientRequest.post(
        `${options.endpoint}/chat/completions`,
      )

      // Apply bearer token header and accept SSE
      const authedReq = pipe(
        baseReq,
        HttpClientRequest.bearerToken(options.authToken),
        HttpClientRequest.setHeader('Accept', 'text/event-stream'),
      )

      // Apply JSON body — bodyJson returns an Effect because it may fail encoding
      const httpRequest = yield* pipe(
        HttpClientRequest.bodyJson(request)(authedReq),
        Effect.mapError(
          (cause) =>
            new DriverError({
              reason: `request_build_failed: ${String(cause)}`,
              status: null,
              body: null,
            }),
        ),
      )

      // Execute the request — get back a Response effect
      const responseEffect = pipe(
        httpClient.execute(httpRequest),
        Effect.mapError(
          (cause) =>
            new DriverError({
              reason: `http_failed: ${String(cause)}`,
              status: null,
              body: null,
            }),
        ),
      )

      // Check HTTP status before streaming — read error body for non-2xx
      const checkedResponseEffect = pipe(
        responseEffect,
        Effect.flatMap((response) => {
          if (response.status >= 200 && response.status < 300) {
            return Effect.succeed(response)
          }
          return pipe(
            response.text,
            Effect.orElse(() => Effect.succeed('')),
            Effect.flatMap((body) =>
              Effect.fail(
                new DriverError({
                  reason: `http_status`,
                  status: response.status,
                  body,
                }),
              ),
            ),
          )
        }),
      )

      // Lift response effect to byte stream
      const byteStream: Stream.Stream<Uint8Array, DriverError> = pipe(
        HttpClientResponse.stream(checkedResponseEffect),
        Stream.mapError((cause) => {
          // Preserve DriverError instances that bubble up from checkedResponseEffect
          if (cause instanceof DriverError) return cause
          return new DriverError({
            reason: `transport_failed: ${String(cause)}`,
            status: null,
            body: null,
          })
        }),
      )

      // bytes → SSE events
      const sseStream = sseChunks(byteStream)

      // SSE events → typed chunks via Schema decode
      const chunkStream: Stream.Stream<ChatCompletionsStreamChunk, DriverError> = pipe(
        sseStream,
        Stream.mapEffect((json) =>
          pipe(
            Schema.decodeUnknown(ChatCompletionsStreamChunk)(json),
            Effect.mapError(
              (cause) =>
                new DriverError({
                  reason: `chunk_decode_failed: ${String(cause)}`,
                  status: null,
                  body: json,
                }),
            ),
          ),
        ),
      )

      return chunkStream
    }),
}
