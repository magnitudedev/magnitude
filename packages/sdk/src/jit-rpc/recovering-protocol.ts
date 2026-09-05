import { RpcClient, RpcClientError, RpcSerialization } from "@effect/rpc"
import type { FromClientEncoded, ResponseExitEncoded } from "@effect/rpc/RpcMessage"
import * as HttpBody from "@effect/platform/HttpBody"
import * as HttpClient from "@effect/platform/HttpClient"
import { Array as Arr, Chunk, Deferred, Effect, Either, Layer, Schema, Scope, Stream } from "effect"
import { JsonValueSchema } from "@magnitudedev/utils/schema"
import type { RecoveringStreamProtocol } from "./recovering-stream-protocol"
import type { AcnRpcRecoveryPolicy } from "@magnitudedev/acn-protocol"
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
  RpcOutcomeUnknown,
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

interface RpcEndpoint {
  readonly id: string
  readonly url: string
}

export interface RecoveringProtocolOptions<InfraError, Endpoint extends RpcEndpoint = RpcEndpoint> {
  readonly endpoint: Effect.Effect<Endpoint, InfraError>
  readonly recover: (
    failedEndpoint: Endpoint,
  ) => Effect.Effect<Endpoint, InfraError>
  readonly rpcPath: string
  readonly streamProtocol: RecoveringStreamProtocol
  readonly isEndpointRetirementExit?: (exit: ResponseExitEncoded["exit"]) => boolean
  // Effect RPC's Protocol.send fixes its failure type to RpcClientError. Keep the
  // adapter here until that upstream boundary supports generic transport errors.
  readonly classifyInfraError: (error: InfraError) => RpcClientError.RpcClientError
  readonly recoveryPolicy: (tag: string) => AcnRpcRecoveryPolicy
}

export const makeRecoveringProtocol = <InfraError, Endpoint extends RpcEndpoint = RpcEndpoint>(
  options: RecoveringProtocolOptions<InfraError, Endpoint>,
) =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const serialization = yield* RpcSerialization.RpcSerialization
    const protocolScope = yield* Scope.Scope

    return yield* Protocol.make(
      Effect.fnUntraced(function* (writeResponse) {
        const send = (
          request: FromClientEncoded,
        ): Effect.Effect<void, RpcClientError.RpcClientError> => {
          if (request._tag !== "Request") return Effect.void

          const isStream = options.streamProtocol.isStream(request.tag)
          const policy = isStream ? undefined : options.recoveryPolicy(request.tag)

          return Effect.suspend(() => {
            let done = false
            let progressed = false
            let consecutiveFailures = 0

            const attempt = (
              endpoint: Endpoint,
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
                  .post(`${endpoint.url}${options.rpcPath}`, {
                    body,
                    headers: { "x-magnitude-acn-id": endpoint.id },
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
                const endpoint = yield* options.endpoint.pipe(
                  Effect.mapError(options.classifyInfraError),
                )
                progressed = false
                yield* Effect.logDebug("jit-rpc attempt").pipe(
                  Effect.annotateLogs({
                    tag: request.tag,
                    id: request.id,
                    url: endpoint.url,
                  }),
                )
                const outcome = yield* Effect.either(attempt(endpoint))
                consecutiveFailures = progressed ? 0 : consecutiveFailures + 1
                if (!done && consecutiveFailures >= 3) {
                  return yield* toRpcClientError(new RecoveryExhausted({ attempts: consecutiveFailures }))
                }
                if (Either.isRight(outcome)) {
                  if (outcome.right._tag === "SubscriptionTerminated") {
                    yield* options.recover(endpoint).pipe(
                      Effect.mapError(options.classifyInfraError),
                    )
                    continue
                  }
                  if (outcome.right._tag === "EndpointRetired") {
                    if (policy === "AtMostOnce") {
                      yield* options.recover(endpoint).pipe(Effect.ignore, Effect.forkIn(protocolScope))
                      return yield* toRpcClientError(new RpcOutcomeUnknown({ tag: request.tag }))
                    }
                    yield* options.recover(endpoint).pipe(
                      Effect.mapError(options.classifyInfraError),
                    )
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
                if (done) return

                if (
                  failure._tag === "RequestEncodeFailed" ||
                  failure._tag === "ResponseDecodeFailed" ||
                  failure._tag === "UnrecognizedMessage" ||
                  (failure._tag === "BadResponseStatus" &&
                    failure.status !== 409 &&
                    failure.status !== 503 &&
                    (failure.status < 500 || failure.status >= 600))
                ) return yield* toRpcClientError(failure)

                const definitelyNotDispatched = failure._tag === "BadResponseStatus" &&
                  (failure.status === 409 || failure.status === 503)
                if (!isStream && policy === "AtMostOnce" && !definitelyNotDispatched) {
                  yield* options.recover(endpoint).pipe(Effect.ignore, Effect.forkIn(protocolScope))
                  return yield* toRpcClientError(new RpcOutcomeUnknown({ tag: request.tag }))
                }
                yield* options.recover(endpoint).pipe(
                  Effect.mapError(options.classifyInfraError),
                )
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

export const recoveringProtocolLayer = <InfraError, Endpoint extends RpcEndpoint = RpcEndpoint>(
  options: RecoveringProtocolOptions<InfraError, Endpoint>,
): Layer.Layer<RpcClient.Protocol, never, HttpClient.HttpClient> =>
  Layer.scoped(
    Protocol,
    makeRecoveringProtocol(options).pipe(Effect.provide(RpcSerialization.layerNdjson)),
  )
