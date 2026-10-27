import { Context, Data, Effect, Layer, Option, Stream } from "effect"
import { IcnClient, type IcnClientService } from "@magnitudedev/icn"
import * as Native from "@magnitudedev/icn-protocol/schemas"
import { type ResponseFailure, ServableModelsUnavailable } from "@magnitudedev/acn-protocol"

/** How a request to the inference service fails: the service's error, an unreachable service, or a request refused before sending. */
type InferenceFailure = Extract<ResponseFailure, { readonly _tag: "inference" | "unreachable" | "rejected" }>

/** A request to the inference service failed. */
export class InferenceFailed extends Data.TaggedError("InferenceFailed")<{
  readonly error: InferenceFailure
}> {}

export interface ChatInference {
  readonly models: Effect.Effect<Native.OpenAiModelsResponse, ServableModelsUnavailable>
  readonly generate: (request: Native.ChatCompletionRequest) => Stream.Stream<Native.ChatCompletionChunk, InferenceFailed>
}
export const ChatInference = Context.GenericTag<ChatInference>("@magnitudedev/acn/ChatInference")

/** The service could not be reached, or its stream broke or was malformed. */
const unreachable = (message: string) => new InferenceFailed({ error: { _tag: "unreachable", message } })

/** The request, as built from the response's settings, is not one the inference API accepts. */
export const requestRejected = new InferenceFailed({
  error: { _tag: "rejected", message: "These settings could not be used for this model." },
})

type ChatCompletion = ReturnType<IcnClientService["chat"]["createChatCompletion"]>
type ClientError =
  | Effect.Effect.Error<ReturnType<IcnClientService["inference"]["listServableModels"]>>
  | Effect.Effect.Error<ChatCompletion>
  | Stream.Stream.Error<Effect.Effect.Success<ChatCompletion>["events"]>

/** The client's error as the failure it is: the one place the chat-completions error object is read. */
export const inferenceFailed = (error: ClientError): InferenceFailed => {
  switch (error._tag) {
    case "GeneratedClientRemoteError": {
      const { code, message, param } = error.body.error
      return new InferenceFailed({
        error: { _tag: "inference", code, message, param: Option.flatMap(param, Option.fromNullable) },
      })
    }
    case "GeneratedClientTransportError":
      return unreachable(
        error.cause instanceof Error && error.cause.message.length > 0
          ? `The inference service could not be reached: ${error.cause.message}`
          : "The inference service could not be reached."
      )
    case "GeneratedClientInvalidResponseError":
      return unreachable(`The inference service sent a malformed response: ${error.message}`)
    case "GeneratedClientIncompleteStreamError":
      return unreachable("The inference stream ended before it was complete.")
    case "GeneratedClientInputError":
      return requestRejected
  }
}

export const ChatInferenceLive = Layer.effect(
  ChatInference,
  Effect.gen(function* () {
    const client = yield* IcnClient
    return ChatInference.of({
      models: client.inference.listServableModels({}).pipe(
        Effect.mapError((error) => new ServableModelsUnavailable({ message: inferenceFailed(error).error.message }))
      ),
      generate: (request) =>
        Stream.unwrap(
          client.chat
            .createChatCompletion({
              payload: request,
              headers: { "Magnitude-Include-Progress": Option.some(true) },
            })
            .pipe(Effect.map((response) => response.events))
        ).pipe(Stream.mapError(inferenceFailed)),
    })
  })
)
