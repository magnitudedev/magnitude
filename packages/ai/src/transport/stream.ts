import { Effect, Stream } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpBody from "@effect/platform/HttpBody"
import type { HttpConnectionFailure, StreamFailure } from "../errors/failure"
import type { AuthApplicator } from "../auth/auth"
import { sseStream } from "./sse"

export interface ExecuteHttpStreamConfig<TWireReq, TWireChunk> {
  readonly url: string
  readonly body: TWireReq
  readonly auth: AuthApplicator
  readonly extraHeaders?: Record<string, string>
  readonly decodePayload: (raw: string) => Effect.Effect<TWireChunk, Error>
  readonly sourceId: string
  readonly doneSignal?: string
}

/**
 * Execute an HTTP POST request expecting an SSE stream response.
 *
 * - Non-2xx responses fail with `HttpConnectionFailure`
 * - Stream read/parse errors are wrapped as `StreamFailure`
 * - Requires `HttpClient.HttpClient` in the environment
 */
export function executeHttpStream<TWireReq, TWireChunk>(
  config: ExecuteHttpStreamConfig<TWireReq, TWireChunk>,
): Effect.Effect<
  Stream.Stream<TWireChunk, StreamFailure>,
  HttpConnectionFailure,
  HttpClient.HttpClient
> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient

    // Build headers via Web API Headers so AuthApplicator works
    const headers = new Headers()
    headers.set("Content-Type", "application/json")
    headers.set("Accept", "text/event-stream")
    if (config.extraHeaders) {
      for (const [k, v] of Object.entries(config.extraHeaders)) {
        headers.set(k, v)
      }
    }
    config.auth(headers)

    // Convert to plain record for HttpClientRequest
    const headerRecord: Record<string, string> = {}
    headers.forEach((value, key) => {
      headerRecord[key] = value
    })

    const request = HttpClientRequest.post(config.url).pipe(
      HttpClientRequest.setHeaders(headerRecord),
      HttpClientRequest.setBody(HttpBody.unsafeJson(config.body)),
    )

    const response = yield* client.execute(request).pipe(
      Effect.mapError((err): HttpConnectionFailure => ({
        status: 0,
        headers: new Headers(),
        body: err.message,
      })),
    )

    if (response.status < 200 || response.status >= 300) {
      const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
      // Extract headers from response
      const respHeaders = new Headers()
      for (const [k, v] of Object.entries(response.headers)) {
        if (v !== undefined) respHeaders.set(k, v)
      }
      return yield* Effect.fail<HttpConnectionFailure>({
        status: response.status,
        headers: respHeaders,
        body,
      })
    }

    // Stream the response body through SSE parsing
    const byteStream: Stream.Stream<Uint8Array, StreamFailure> = response.stream.pipe(
      Stream.mapError((err): StreamFailure => ({
        _tag: "ReadFailure",
        cause: new Error(err.message),
      })),
    )

    // Wrap decodePayload to produce StreamFailure on error
    const wrappedDecode = (raw: string): Effect.Effect<TWireChunk, StreamFailure> =>
      config.decodePayload(raw).pipe(
        Effect.mapError((cause): StreamFailure => ({
          _tag: "ChunkDecodeFailure",
          payload: raw,
          cause,
        })),
      )

    return sseStream(byteStream, wrappedDecode, config.doneSignal)
  })
}
