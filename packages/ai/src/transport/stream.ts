import { Clock, Duration, Effect, Stream } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpBody from "@effect/platform/HttpBody"
import type * as HttpClientError from "@effect/platform/HttpClientError"
import type {
  AcceptedHttpResponse,
  ProviderCall,
  RejectedHttpResponse,
  StreamStartFailure,
  StreamFailure,
  LastStreamActivity,
  StreamProgress,
} from "../errors/failure"
import {
  acceptedHttpResponse,
  headersFromHeaderList,
  payloadSample,
  rejectedHttpResponse,
  causeInfoText,
  StreamOperationalFailure,
  StreamProviderCorrectnessViolation,
  StreamStartClientCorrectnessViolation,
  StreamStartOperationalFailure,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
  toCauseInfo,
} from "../errors/failure"
import { streamStartFailureFromRejectedResponse } from "../errors/classify"
import type { AuthApplicator } from "../auth/auth"
import { sseStream } from "./sse"

const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 60_000

export interface ExecuteHttpStreamConfig<
  TWireReq,
  TWireChunk,
> {
  readonly call: ProviderCall
  readonly body: TWireReq
  readonly auth: AuthApplicator
  readonly extraHeaders?: Record<string, string>
  readonly decodePayload: (raw: string) => Effect.Effect<TWireChunk, unknown>
  readonly classifyRejectedResponse?: (
    call: ProviderCall,
    response: RejectedHttpResponse,
  ) => StreamStartProviderRejection | StreamStartProviderCorrectnessViolation
  readonly doneSignal?: string
  readonly idleTimeoutMs?: number
}

function toHeaderRecord(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {}
  headers.forEach((value, key) => {
    result[key] = value
  })
  return result
}

function progress(dataPayloadsDecoded: number): StreamProgress {
  return { dataPayloadsDecoded, modelEventsEmitted: 0 }
}

function bodyReadFailure(
  err: HttpClientError.ResponseError,
  call: ProviderCall,
  response: AcceptedHttpResponse,
  dataPayloadsDecoded: number,
): StreamFailure {
  return new StreamOperationalFailure({
    call,
    response,
    reason: {
      _tag: "BodyReadFailure",
      readError: {
        _tag: "EffectResponseBodyError",
        effectReason: err.reason,
        cause: toCauseInfo(err.cause ?? err),
      },
    },
    progress: progress(dataPayloadsDecoded),
  })
}

function chunkDecodeFailure(
  raw: string,
  cause: unknown,
  call: ProviderCall,
  response: AcceptedHttpResponse,
  dataPayloadsDecoded: number,
): StreamFailure {
  const sample = payloadSample(raw)
  const message = causeMessage(cause)
  if (isJsonParseCause(cause)) {
    return new StreamProviderCorrectnessViolation({
      call,
      response,
      violation: {
        _tag: "InvalidProviderChunk",
        problem: {
          _tag: "InvalidJson",
          payload: sample,
          cause: toCauseInfo(cause),
        },
      },
      progress: progress(dataPayloadsDecoded),
    })
  }

  return new StreamProviderCorrectnessViolation({
    call,
    response,
    violation: {
      _tag: "InvalidProviderChunk",
      problem: {
        _tag: "InvalidChunkSchema",
        payload: sample,
        issue: { message },
        cause: toCauseInfo(cause),
      },
    },
    progress: progress(dataPayloadsDecoded),
  })
}

function isJsonParseCause(cause: unknown): boolean {
  return typeof cause === "object"
    && cause !== null
    && "_tag" in cause
    && (cause as { readonly _tag: unknown })._tag === "ChatPayloadJsonParseError"
}

function causeMessage(cause: unknown): string {
  if (cause instanceof Error && cause.message.trim().length > 0) return cause.message
  if (typeof cause === "object" && cause !== null && "message" in cause) {
    const message = (cause as { readonly message: unknown }).message
    if (typeof message === "string" && message.trim().length > 0) return message
  }
  return String(cause)
}

export interface HttpStreamResult<TWireChunk> {
  readonly stream: Stream.Stream<TWireChunk, StreamFailure>
  readonly responseHeaders: Headers
  readonly call: ProviderCall
  readonly response: AcceptedHttpResponse
}

/**
 * Execute an HTTP POST request expecting an SSE stream response.
 *
 * - startup failures are `StreamStartFailure`
 * - after a 2xx response, body/read/decode/stall failures are `StreamFailure`
 * - successful startup returns the stream plus the accepted response facts
 */
export function executeHttpStream<
  TWireReq,
  TWireChunk,
>(
  config: ExecuteHttpStreamConfig<TWireReq, TWireChunk>,
): Effect.Effect<
  HttpStreamResult<TWireChunk>,
  StreamStartFailure,
  HttpClient.HttpClient
> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient

    const headers = new Headers()
    headers.set("Content-Type", "application/json")
    headers.set("Accept", "text/event-stream")
    if (config.extraHeaders) {
      for (const [k, v] of Object.entries(config.extraHeaders)) {
        headers.set(k, v)
      }
    }

    yield* Effect.try({
      try: () => config.auth(headers),
      catch: (cause) => {
        const causeInfo = toCauseInfo(cause)
        return new StreamStartClientCorrectnessViolation({
          call: config.call,
          component: "auth_applicator",
          message: `Could not apply model authentication: ${causeInfoText(causeInfo)}`,
          evidence: { _tag: "AuthApplicationFailed", cause: causeInfo },
        })
      },
    })

    const request = yield* Effect.try({
      try: () =>
        HttpClientRequest.post(config.call.url).pipe(
          HttpClientRequest.setHeaders(toHeaderRecord(headers)),
          HttpClientRequest.setBody(HttpBody.unsafeJson(config.body)),
        ),
      catch: (cause) => {
        const causeInfo = toCauseInfo(cause)
        return new StreamStartClientCorrectnessViolation({
          call: config.call,
          component: "request_body_encoder",
          message: `Could not encode model request body: ${causeInfoText(causeInfo)}`,
          evidence: { _tag: "RequestBodyEncodingFailed", cause: causeInfo },
        })
      },
    })

    const rawResponse = yield* client.execute(request).pipe(
      Effect.mapError((err) => new StreamStartOperationalFailure({
        call: config.call,
        reason: { _tag: "RequestFailedBeforeResponse", cause: toCauseInfo(err) },
      })),
    )

    if (rawResponse.status < 200 || rawResponse.status >= 300) {
      const body = yield* rawResponse.text.pipe(Effect.orElseSucceed(() => ""))
      const classifyRejectedResponse =
        config.classifyRejectedResponse ?? streamStartFailureFromRejectedResponse
      return yield* classifyRejectedResponse(
        config.call,
        rejectedHttpResponse(rawResponse.status, rawResponse.headers, body),
      ) as StreamStartFailure
    }

    const response = acceptedHttpResponse(rawResponse.status, rawResponse.headers)
    const responseHeaders = headersFromHeaderList(response.headers)
    let dataPayloadsDecoded = 0
    let lastActivity: LastStreamActivity = { _tag: "NoActivity" }
    const idleTimeoutMs = config.idleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS

    const now = yield* Clock.currentTimeMillis
    lastActivity = { _tag: "ResponseAccepted", atEpochMs: now }

    const byteStream: Stream.Stream<Uint8Array, StreamFailure> = rawResponse.stream.pipe(
      Stream.tap(() =>
        Effect.gen(function* () {
          const now = yield* Clock.currentTimeMillis
          lastActivity = { _tag: "BodyBytesRead", atEpochMs: now }
        }),
      ),
      Stream.tapError((err) =>
        Effect.logError("[stream] Response body read failure", {
          failure: bodyReadFailure(err, config.call, response, dataPayloadsDecoded),
        }),
      ),
      Stream.mapError((err) => bodyReadFailure(err, config.call, response, dataPayloadsDecoded)),
      Stream.timeoutFail(
        () => new StreamOperationalFailure({
          call: config.call,
          response,
          reason: { _tag: "StallTimeout", timeoutMs: idleTimeoutMs, lastActivity },
          progress: progress(dataPayloadsDecoded),
        }),
        Duration.millis(idleTimeoutMs),
      ),
    )

    const wrappedDecode = (raw: string): Effect.Effect<TWireChunk, StreamFailure> =>
      config.decodePayload(raw).pipe(
        Effect.tap(() =>
          Effect.gen(function* () {
            dataPayloadsDecoded += 1
            const now = yield* Clock.currentTimeMillis
            lastActivity = { _tag: "DataPayloadDecoded", atEpochMs: now }
          }),
        ),
        Effect.tapError((cause) =>
          Effect.logError("[stream] Chunk decode failure", {
            payload: raw,
            error: String(cause),
          }),
        ),
        Effect.mapError((cause) => chunkDecodeFailure(raw, cause, config.call, response, dataPayloadsDecoded)),
      )

    return {
      stream: sseStream(byteStream, wrappedDecode, config.doneSignal),
      responseHeaders,
      call: config.call,
      response,
    }
  })
}
