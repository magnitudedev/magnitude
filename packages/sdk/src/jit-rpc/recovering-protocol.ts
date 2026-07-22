import { RpcClient, RpcClientError, RpcSerialization } from "@effect/rpc"
import type { FromClientEncoded, ResponseExitEncoded } from "@effect/rpc/RpcMessage"
import * as HttpBody from "@effect/platform/HttpBody"
import * as HttpClient from "@effect/platform/HttpClient"
import { Array as Arr, Chunk, Deferred, Effect, Either, Layer, Schema, Stream } from "effect"
import { JsonValueSchema } from "@magnitudedev/utils/schema"
import type { JitDaemonCoordinator } from "./daemon-resolver"
import type { RecoveringStreamProtocol } from "./recovering-stream-protocol"
import {
  type JitRpcAttemptFailure,
  toRpcClientError,
  RequestEncodeFailed,
  TransportRequestFailed,
  BadResponseStatus,
  ResponseDecodeFailed,
  UnrecognizedMessage,
  SubscriptionProtocolViolation,
  StreamLivenessTimeout,
  StreamEndedWithoutExit,
  RecoveryExhausted,
} from "./errors"
import { isChunkMessage, isFromServerEncoded, isTerminalMessage } from "./transport"

const { Protocol } = RpcClient

type AttemptOutcome =
  | { readonly _tag: "Completed" }
  | { readonly _tag: "SubscriptionTerminated" }
  | { readonly _tag: "EndpointRetired" }

const Completed: AttemptOutcome = { _tag: "Completed" }
const SubscriptionTerminated: AttemptOutcome = { _tag: "SubscriptionTerminated" }
const EndpointRetired: AttemptOutcome = { _tag: "EndpointRetired" }

export interface RecoveringProtocolOptions<InfraError> {
  readonly coordinator: JitDaemonCoordinator<InfraError>
  readonly rpcPath: string
  readonly streamProtocol: RecoveringStreamProtocol
  readonly isEndpointRetirementExit?: (exit: ResponseExitEncoded["exit"]) => boolean
  readonly classifyInfraError: (error: InfraError) => RpcClientError.RpcClientError
}

export const makeRecoveringProtocol = <InfraError>(
  options: RecoveringProtocolOptions<InfraError>,
) =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const serialization = yield* RpcSerialization.RpcSerialization

    return yield* Protocol.make(
      Effect.fnUntraced(function* (writeResponse) {
        const send = (
          request: FromClientEncoded,
        ): Effect.Effect<void, RpcClientError.RpcClientError> => {
          if (request._tag !== "Request") return Effect.void

          const isStream = options.streamProtocol.isStream(request.tag)

          return Effect.suspend(() => {
            let done = false
            let progressed = false
            let failuresWithoutProgress = 0

            const attempt = (
              endpointUrl: string,
            ): Effect.Effect<AttemptOutcome, JitRpcAttemptFailure> =>
              Effect.gen(function* () {
                const lifecycle = yield* Deferred.make<AttemptOutcome>()
                let lifecycleObserved = false
                const parser = serialization.unsafeMake()
                const encoded = yield* Effect.try({
                  try: () => parser.encode(request),
                  catch: () =>
                    new RequestEncodeFailed({
                      message: "Failed to encode request",
                    }),
                })
                if (encoded === undefined) {
                  return yield* new RequestEncodeFailed({
                    message: "Serialization produced no request payload",
                  })
                }

                const body =
                  typeof encoded === "string"
                    ? HttpBody.text(encoded, serialization.contentType)
                    : HttpBody.uint8Array(encoded, serialization.contentType)

                const response = yield* client
                  .post(`${endpointUrl}${options.rpcPath}`, {
                    body,
                  })
                  .pipe(
                    Effect.mapError(
                      () =>
                        new TransportRequestFailed({
                          message: "Failed to send request to daemon",
                        }),
                    ),
                  )
                if (response.status < 200 || response.status >= 300) {
                  return yield* new BadResponseStatus({
                    status: response.status,
                  })
                }

                const handleChunk = (
                  chunk: Chunk.Chunk<Uint8Array>,
                ): Effect.Effect<void, JitRpcAttemptFailure> =>
                  Effect.gen(function* () {
                    if (lifecycleObserved) return
                    const messages = yield* Effect.try({
                      try: () =>
                        Chunk.toReadonlyArray(chunk).flatMap((bytes) => parser.decode(bytes)),
                      catch: () =>
                        new ResponseDecodeFailed({
                          message: "Failed to decode daemon response",
                        }),
                    })
                    for (const message of messages) {
                      if (!isFromServerEncoded(message)) {
                        return yield* new UnrecognizedMessage({
                          message: "Daemon sent an unrecognized response message",
                        })
                      }
                      if (isChunkMessage(message)) {
                        if (!isStream) {
                          progressed = true
                          yield* writeResponse(message)
                          continue
                        }

                        const jsonValues = yield* Schema.decodeUnknown(
                          Schema.Array(JsonValueSchema),
                        )(message.values).pipe(
                          Effect.mapError(
                            () =>
                              new ResponseDecodeFailed({
                                message: "Daemon stream chunk was not valid JSON",
                              }),
                          ),
                        )
                        const decoded = yield* options.streamProtocol.decodeChunk(jsonValues).pipe(
                          Effect.mapError(
                            () =>
                              new SubscriptionProtocolViolation({
                                message: "Daemon stream violated its wire protocol",
                              }),
                          ),
                        )
                        switch (decoded._tag) {
                          case "Terminated":
                            lifecycleObserved = true
                            yield* Deferred.succeed(lifecycle, SubscriptionTerminated)
                            return
                          case "Continue":
                            progressed = progressed || decoded.progressed
                            if (Arr.isNonEmptyReadonlyArray(decoded.values)) {
                              yield* writeResponse({ ...message, values: decoded.values })
                            }
                            break
                        }
                        continue
                      }
                      if (
                        message._tag === "Exit" &&
                        isStream &&
                        options.streamProtocol.isExitWithoutTermination(message.exit)
                      ) {
                        return yield* new SubscriptionProtocolViolation({
                          message: "Subscription exited without its terminal control",
                        })
                      }
                      if (
                        message._tag === "Exit" &&
                        !isStream &&
                        options.isEndpointRetirementExit?.(message.exit) === true
                      ) {
                        lifecycleObserved = true
                        yield* Deferred.succeed(lifecycle, EndpointRetired)
                        return
                      }
                      if (isTerminalMessage(message)) done = true
                      yield* writeResponse(message)
                    }
                  })

                const byteStream = isStream
                  ? response.stream.pipe(
                      Stream.mapError(
                        () =>
                          new TransportRequestFailed({
                            message: "Failed to reach daemon",
                          }),
                      ),
                      Stream.timeoutFail(
                        () =>
                          new StreamLivenessTimeout({
                            message: "Daemon stream silent past liveness timeout",
                          }),
                        `${options.streamProtocol.livenessTimeoutMs} millis`,
                      ),
                    )
                  : response.stream.pipe(
                      Stream.mapError(
                        () =>
                          new TransportRequestFailed({
                            message: "Failed to reach daemon",
                          }),
                      ),
                    )

                const consume = Stream.runForEachChunk(byteStream, handleChunk).pipe(
                  Effect.flatMap(() =>
                    done
                      ? Effect.succeed(Completed)
                      : Effect.fail(new StreamEndedWithoutExit({
                          message: "Daemon response ended without an exit",
                        })),
                  ),
                )

                return yield* Effect.raceFirst(Deferred.await(lifecycle), consume)
              })

            return Effect.gen(function* () {
              while (!done) {
                const lease = yield* options.coordinator.ensure.pipe(
                  Effect.mapError(options.classifyInfraError),
                )
                const endpoint = lease.endpoint
                progressed = false
                yield* Effect.logDebug("jit-rpc attempt").pipe(
                  Effect.annotateLogs({
                    tag: request.tag,
                    id: request.id,
                    url: endpoint.url,
                  }),
                )
                const outcome = yield* Effect.either(attempt(endpoint.url))
                if (Either.isRight(outcome)) {
                  if (outcome.right._tag === "SubscriptionTerminated") {
                    yield* options.coordinator.awaitSuccessor(lease).pipe(
                      Effect.mapError(options.classifyInfraError),
                    )
                    continue
                  }
                  if (outcome.right._tag === "EndpointRetired") {
                    yield* options.coordinator.invalidate(lease, {
                      awaitDifferentEndpoint: true,
                    })
                    continue
                  }
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
                if (failure._tag === "SubscriptionProtocolViolation") {
                  return yield* toRpcClientError(failure)
                }
                yield* options.coordinator.invalidate(lease)
                if (done) return

                failuresWithoutProgress = progressed ? 1 : failuresWithoutProgress + 1
                if (failuresWithoutProgress >= 2) {
                  yield* Effect.logDebug("jit-rpc surfacing fatal").pipe(
                    Effect.annotateLogs({
                      tag: request.tag,
                      id: request.id,
                      failures: failuresWithoutProgress,
                    }),
                  )
                  return yield* toRpcClientError(
                    new RecoveryExhausted({
                      attempts: failuresWithoutProgress,
                    }),
                  )
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
      }),
    )
  })

export const recoveringProtocolLayer = <InfraError>(
  options: RecoveringProtocolOptions<InfraError>,
): Layer.Layer<RpcClient.Protocol, never, HttpClient.HttpClient> =>
  Layer.scoped(
    Protocol,
    makeRecoveringProtocol(options).pipe(Effect.provide(RpcSerialization.layerNdjson)),
  )
