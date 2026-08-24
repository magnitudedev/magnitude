import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as HttpClient from "@effect/platform/HttpClient"
import type { GeneratedClientOptions } from "@magnitudedev/openapi-effect/client-runtime"
import type { GeneratedClientError } from "@magnitudedev/openapi-effect/client-runtime"
import { Client, Mutation, QueryClient, Subscription } from "@magnitudedev/effect-query"
import { makeIcnApiClient, type IcnApiClient } from "@magnitudedev/icn-protocol/client"
import * as S from "@magnitudedev/icn-protocol/schemas"
import { Effect, Option, Stream } from "effect"
import {
  Inference,
  inferenceImplementationsLayer,
  type InferenceClientError,
} from "./inference"
import { MAGNITUDE_INFERENCE_BASE_URL } from "./inference-endpoint"

export type InferenceEffectQueryClient = Client.GroupClient<
  typeof Inference,
  import("@magnitudedev/effect-query").Operation.Implementations<InferenceClientError>,
  never
>

export interface InferenceClient {
  /** The generated OpenAPI transport for infrastructure and non-product consumers. */
  readonly transport: IcnApiClient
  readonly query: <Input, Data, Error, Requirements>(
    definition: import("@magnitudedev/effect-query").Query.Query<Input, Data, Error, Requirements>,
    input: Input,
  ) => Effect.Effect<Data, Error | InferenceClientError | QueryClient.QueryBatchError>
  readonly mutate: <Input, Output, Error, Requirements, SynchronizationError>(
    definition: import("@magnitudedev/effect-query").Mutation.Mutation<
      Input,
      Output,
      Error,
      Requirements,
      SynchronizationError
    >,
    input: Input,
  ) => Effect.Effect<
    Output,
    Error | InferenceClientError | Mutation.MutationSynchronizationError<Output, SynchronizationError>
  >
  readonly subscribe: <Input, Event, Error, Requirements>(
    definition: import("@magnitudedev/effect-query").Subscription.Subscription<
      Input,
      Event,
      Error,
      Requirements
    >,
    input: Input,
  ) => Stream.Stream<Event, Error | InferenceClientError>
  /** Low-level streaming inference with a consumer-friendly progress option. */
  readonly streamChatCompletion: (
    payload: typeof S.ChatCompletionRequest.Type,
    options?: { readonly includeProgress?: boolean },
  ) => Stream.Stream<typeof S.ChatCompletionChunk.Type, GeneratedClientError>
  readonly streamResponse: (
    payload: typeof S.ResponseCreateRequest.Type,
    options?: { readonly includeProgress?: boolean },
  ) => Stream.Stream<typeof S.ResponseStreamEvent.Type, GeneratedClientError>
  readonly close: Effect.Effect<void>
}

/**
 * Makes one standalone inference client over the same authored Effect Query
 * boundary used by first-party connected clients. It owns one cache/registry;
 * callers should reuse it for the lifetime of their CLI process or SDK client.
 */
export const makeInferenceClient = (
  options: GeneratedClientOptions = {
    baseUrl: MAGNITUDE_INFERENCE_BASE_URL,
  },
): Effect.Effect<InferenceClient, never, HttpClient.HttpClient> => Effect.gen(function* () {
  const transport = yield* makeIcnApiClient(options)
  const client: InferenceEffectQueryClient = Client.make(
    Inference,
    inferenceImplementationsLayer(transport),
  )
  const registry = AtomRegistry.make()
  const query = <Input, Data, Error, Requirements>(
    definition: import("@magnitudedev/effect-query").Query.Query<Input, Data, Error, Requirements>,
    input: Input,
  ) => AtomRegistry.getResult(
    registry,
    client.runtime.atom(QueryClient.QueryClient.pipe(
      Effect.flatMap((queryClient) => queryClient.refetch(definition.match(input))),
      Effect.zipRight(QueryClient.fetch(definition, input)),
    )),
  ) as Effect.Effect<Data, Error | InferenceClientError | QueryClient.QueryBatchError>
  const mutate = <Input, Output, Error, Requirements, SynchronizationError>(
    definition: import("@magnitudedev/effect-query").Mutation.Mutation<
      Input,
      Output,
      Error,
      Requirements,
      SynchronizationError
    >,
    input: Input,
  ) => AtomRegistry.getResult(
    registry,
    client.runtime.atom(Mutation.execute(client.mutation(definition as never), input).pipe(
      Effect.provideService(AtomRegistry.AtomRegistry, registry),
    )),
  ) as Effect.Effect<
    Output,
    Error | InferenceClientError | Mutation.MutationSynchronizationError<Output, SynchronizationError>
  >
  const subscribe = <Input, Event, Error, Requirements>(
    definition: import("@magnitudedev/effect-query").Subscription.Subscription<
      Input,
      Event,
      Error,
      Requirements
    >,
    input: Input,
  ) => Subscription.events(client.subscription(definition as never, input as never)).pipe(
    Stream.provideService(AtomRegistry.AtomRegistry, registry),
  ) as Stream.Stream<Event, Error | InferenceClientError>
  const progressHeaders = (includeProgress: boolean | undefined) => ({
    "Magnitude-Include-Progress": Option.fromNullable(includeProgress),
  })
  const streamChatCompletion = (
    payload: typeof S.ChatCompletionRequest.Type,
    streamOptions?: { readonly includeProgress?: boolean },
  ) => Stream.unwrap(Effect.map(transport.chat.createChatCompletion({
    payload,
    headers: progressHeaders(streamOptions?.includeProgress),
  }), ({ events }) => events))
  const streamResponse = (
    payload: typeof S.ResponseCreateRequest.Type,
    streamOptions?: { readonly includeProgress?: boolean },
  ) => Stream.unwrap(Effect.map(transport.inference.createResponse({
    payload,
    headers: progressHeaders(streamOptions?.includeProgress),
  }), ({ events }) => events))
  return {
    transport,
    query,
    mutate,
    subscribe,
    streamChatCompletion,
    streamResponse,
    close: Effect.sync(() => registry.dispose()),
  }
})
