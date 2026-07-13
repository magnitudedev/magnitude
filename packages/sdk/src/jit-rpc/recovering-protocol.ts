import { RpcClient, RpcClientError, RpcSerialization } from "@effect/rpc"
import type { FromClientEncoded } from "@effect/rpc/RpcMessage"
import * as HttpBody from "@effect/platform/HttpBody"
import * as HttpClient from "@effect/platform/HttpClient"
import { Array as Arr, Chunk, Effect, Either, Layer, Stream } from "effect"
import type { JitDaemonResolver } from "./daemon-resolver"
import type { ResidentStreamPolicy } from "./resident-streams"
import {
  type JitRpcTransportError,
  toRpcClientError,
  RequestEncodeFailed,
  TransportRequestFailed,
  BadResponseStatus,
  ResponseDecodeFailed,
  UnrecognizedMessage,
  ResidentStreamRelinquished,
  StreamLivenessTimeout,
  StreamEndedWithoutExit,
  TransportExhausted,
} from "./errors"
import { isChunkMessage, isFromServerEncoded, isTerminalMessage } from "./transport"

const { Protocol } = RpcClient

export interface RecoveringProtocolOptions<InfraError> {
  readonly resolver: JitDaemonResolver<InfraError>
  readonly rpcPath: string
  readonly streamPolicy: ResidentStreamPolicy
  readonly classifyInfraError: (error: InfraError) => RpcClientError.RpcClientError
}

export const makeRecoveringProtocol = <InfraError>(
  options: RecoveringProtocolOptions<InfraError>,
) =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const serialization = yield* RpcSerialization.RpcSerialization

    return yield* Protocol.make(Effect.fnUntraced(function* (writeResponse) {
      const send = (request: FromClientEncoded): Effect.Effect<void, RpcClientError.RpcClientError> => {
        if (request._tag !== "Request") return Effect.void

        const isResident = options.streamPolicy.isResident(request.tag)

        return Effect.suspend(() => {
          let done = false
          let progressed = false
          let failuresWithoutProgress = 0

          const attempt = (endpointUrl: string): Effect.Effect<void, JitRpcTransportError> =>
            Effect.gen(function* () {
              const parser = serialization.unsafeMake()
              const encoded = yield* Effect.try({
                try: () => parser.encode(request),
                catch: () => new RequestEncodeFailed({ message: "Failed to encode request" }),
              })
              if (encoded === undefined) {
                return yield* new RequestEncodeFailed({ message: "Serialization produced no request payload" })
              }

              const body = typeof encoded === "string"
                ? HttpBody.text(encoded, serialization.contentType)
                : HttpBody.uint8Array(encoded, serialization.contentType)

              const response = yield* client.post(`${endpointUrl}${options.rpcPath}`, { body }).pipe(
                Effect.mapError(() => new TransportRequestFailed({ message: "Failed to send request to daemon" })),
              )
              if (response.status < 200 || response.status >= 300) {
                return yield* new BadResponseStatus({ status: response.status })
              }

              const handleChunk = (chunk: Chunk.Chunk<Uint8Array>): Effect.Effect<void, JitRpcTransportError> =>
                Effect.gen(function* () {
                  const messages = yield* Effect.try({
                    try: () => Chunk.toReadonlyArray(chunk).flatMap((bytes) => parser.decode(bytes)),
                    catch: () => new ResponseDecodeFailed({ message: "Failed to decode daemon response" }),
                  })
                  progressed = true
                  for (const message of messages) {
                    if (!isFromServerEncoded(message)) {
                      return yield* new UnrecognizedMessage({ message: "Daemon sent an unrecognized response message" })
                    }
                    if (isChunkMessage(message)) {
                      const values = message.values.filter((value) => !options.streamPolicy.isHeartbeatChunk(value))
                      if (!Arr.isNonEmptyReadonlyArray(values)) continue
                      yield* writeResponse({ ...message, values })
                      continue
                    }
                    if (isResident && message._tag === "Exit" && options.streamPolicy.isRelinquishExit(message.exit)) {
                      return yield* new ResidentStreamRelinquished({ message: "Daemon relinquished a resident stream" })
                    }
                    if (isTerminalMessage(message)) done = true
                    yield* writeResponse(message)
                  }
                })

              const byteStream = isResident
                ? response.stream.pipe(
                    Stream.mapError(() => new TransportRequestFailed({ message: "Failed to reach daemon" })),
                    Stream.timeoutFail(
                      () => new StreamLivenessTimeout({ message: "Daemon stream silent past liveness timeout" }),
                      `${options.streamPolicy.livenessTimeoutMs} millis`,
                    ),
                  )
                : response.stream.pipe(
                    Stream.mapError(() => new TransportRequestFailed({ message: "Failed to reach daemon" })),
                  )

              yield* Stream.runForEachChunk(byteStream, handleChunk)

              if (!done) {
                return yield* new StreamEndedWithoutExit({ message: "Daemon response ended without an exit" })
              }
            })

          return Effect.gen(function* () {
            while (!done) {
              const endpoint = yield* options.resolver.resolve.pipe(
                Effect.mapError(options.classifyInfraError),
              )
              progressed = false
              yield* Effect.logDebug("jit-rpc attempt").pipe(
                Effect.annotateLogs({ tag: request.tag, id: request.id, url: endpoint.url }),
              )
              const outcome = yield* Effect.either(attempt(endpoint.url))
              if (Either.isRight(outcome)) {
                yield* Effect.logDebug("jit-rpc completed").pipe(
                  Effect.annotateLogs({ tag: request.tag, id: request.id }),
                )
                return
              }

              const failure = outcome.left
              yield* Effect.logDebug("jit-rpc attempt failed").pipe(
                Effect.annotateLogs({
                  tag: request.tag,
                  id: request.id,
                  progressed,
                  done,
                  error: failure._tag,
                }),
              )
              yield* options.resolver.invalidate(endpoint)
              if (done) return

              failuresWithoutProgress = progressed ? 1 : failuresWithoutProgress + 1
              if (failuresWithoutProgress >= 2) {
                yield* Effect.logDebug("jit-rpc surfacing fatal").pipe(
                  Effect.annotateLogs({ tag: request.tag, id: request.id, failures: failuresWithoutProgress }),
                )
                return yield* toRpcClientError(new TransportExhausted({ attempts: failuresWithoutProgress }))
              }
            }
          })
        })
      }

      return {
        send,
        supportsAck: false,
        supportsTransferables: false,
      }
    }))
  })

export const recoveringProtocolLayer = <InfraError>(
  options: RecoveringProtocolOptions<InfraError>,
): Layer.Layer<RpcClient.Protocol, never, HttpClient.HttpClient> =>
  Layer.scoped(Protocol, makeRecoveringProtocol(options).pipe(
    Effect.provide(RpcSerialization.layerNdjson),
  ))
