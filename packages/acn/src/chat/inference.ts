import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { IcnClient } from "@magnitudedev/icn"
import * as Native from "@magnitudedev/icn-protocol/schemas"
import { ChatFailed } from "@magnitudedev/acn-protocol"

export interface ChatInference {
  readonly models: Effect.Effect<Native.OpenAiModelsResponse, ChatFailed>
  readonly generate: (
    request: Native.ChatCompletionRequest
  ) => Stream.Stream<Native.ChatCompletionChunk, ChatFailed>
}
export const ChatInference = Context.GenericTag<ChatInference>(
  "@magnitudedev/acn/ChatInference"
)
export const chatInferenceError = (error: unknown): ChatFailed => {
  if (
    error &&
    typeof error === "object" &&
    "_tag" in error &&
    error._tag === "GeneratedClientRemoteError" &&
    "body" in error
  ) {
    const decoded = Schema.decodeUnknownOption(
      Schema.Struct({ error: Schema.Struct({ message: Schema.String }) })
    )(error.body)
    if (Option.isSome(decoded))
      return new ChatFailed({
        code: "Inference",
        message: decoded.value.error.message,
      })
  }
  return new ChatFailed({
    code: "Inference",
    message:
      error instanceof Error
        ? error.message
        : "The inference service could not complete this request. Check Status and try again.",
  })
}
export const ChatInferenceLive = Layer.effect(
  ChatInference,
  Effect.gen(function* () {
    const client = yield* IcnClient
    return ChatInference.of({
      models: client.inference
        .listServableModels({})
        .pipe(Effect.mapError(chatInferenceError)),
      generate: (request) =>
        Stream.unwrap(
          client.chat
            .createChatCompletion({
              payload: request,
              headers: { "Magnitude-Include-Progress": Option.some(true) },
            })
            .pipe(Effect.map((response) => response.events))
        ).pipe(Stream.mapError(chatInferenceError)),
    })
  })
)
