import { Cause, Context, Effect, Layer, Match, Option, Schema, Stream } from "effect"
import {
  ModelCatalogError,
  ModelDiscoveryOperationIdSchema,
  ProviderModelIdSchema,
  StreamClientCorrectnessViolation,
  StreamOperationalFailure,
  StreamProviderCorrectnessViolation,
  StreamProviderError,
  StreamStartClientCorrectnessViolation,
  StreamStartOperationalFailure,
  StreamStartProviderCorrectnessViolation,
  acceptedHttpResponse,
  payloadSample,
  rejectedHttpResponse,
  streamStartFailureFromRejectedResponse,
  toCauseInfo,
  nativeChatCompletionsCodec,
  type BaseCallOptions,
  ChatCompletionsStreamChunk,
  type ProviderModelBindOptions,
  type ProviderId,
  type ProviderModelId,
  type StreamFailure,
  type StreamStartFailure,
  type Prompt,
  type ToolCallId,
  type ToolDefinition,
} from "@magnitudedev/ai"
import { IcnClient, type IcnClientService } from "../client.js"
import * as Generated from "../generated/schemas.js"
import {
  GeneratedClientInvalidResponseError,
  type GeneratedClientError,
} from "@magnitudedev/openapi-effect/client-runtime"
import { IcnInventory } from "../inventory/index.js"
import { IcnRecipes } from "../recipes/service.js"
import type { LocalProviderSource } from "./provider.js"
import {
  NativeIcnModelIdSchema,
  candidateLocalModelId,
  localProviderModelId,
  nativeLocalModelId,
} from "./model-identity.js"

const catalogError = <Cause>(
  message: string,
  ...cause: readonly [] | readonly [Cause]
) => Option.match(Option.fromIterable(cause), {
  onNone: () => new ModelCatalogError({ message }),
  onSome: (value) => new ModelCatalogError({ message, cause: value }),
})

type IcnClientError = GeneratedClientError<Generated.ErrorResponse>

const generatedClientErrorMessage = (error: IcnClientError): string => Match.value(error).pipe(
  Match.tag("GeneratedClientRemoteError", (remote) => remote.body.error.message),
  Match.tag("GeneratedClientTransportError", (transport) =>
    Cause.pretty(Cause.fail(transport.cause))),
  Match.tag("GeneratedClientInputError", (input) =>
    `Invalid ICN request input at ${input.location}`),
  Match.tag("GeneratedClientInvalidResponseError", (invalid) => invalid.message),
  Match.tag("GeneratedClientIncompleteStreamError", (incomplete) =>
    `ICN stream ended without a terminal event (${incomplete.termination})`),
  Match.exhaustive,
)

const generatedStartFailure = (
  call: { provider: string; model: string; method: "POST"; url: string },
  error: IcnClientError,
): StreamStartFailure => Match.value(error).pipe(
  Match.tag("GeneratedClientRemoteError", (remote) => streamStartFailureFromRejectedResponse(
    call,
    rejectedHttpResponse(remote.status, remote.headers, JSON.stringify(remote.body)),
  )),
  Match.tag("GeneratedClientTransportError", (transport) => new StreamStartOperationalFailure({
    call,
    reason: { _tag: "RequestFailedBeforeResponse", cause: toCauseInfo(transport.cause) },
  })),
  Match.tag("GeneratedClientInputError", (input) => new StreamStartClientCorrectnessViolation({
    call,
    component: "request_builder",
    message: `Generated ICN request input was invalid at ${input.location}`,
    evidence: { _tag: "RequestBodyEncodingFailed", cause: toCauseInfo(input.cause) },
  })),
  Match.tag("GeneratedClientInvalidResponseError", (invalid) => new StreamStartProviderCorrectnessViolation({
    call,
    response: null,
    violation: {
      _tag: "UnexpectedResponseShape",
      status: invalid.status,
      body: payloadSample(""),
      issue: { message: invalid.message },
    },
  })),
  Match.tag("GeneratedClientIncompleteStreamError", (incomplete) => new StreamStartClientCorrectnessViolation({
    call,
    component: "request_builder",
    message: "Generated ICN client reported an incomplete stream before admission",
    evidence: {
      _tag: "UnexpectedDefectCaught",
      cause: { _tag: "ErrorCause", name: incomplete._tag, message: incomplete.termination },
    },
  })),
  Match.exhaustive,
)

const generatedBodyFailure = (
  call: { provider: string; model: string; method: "POST"; url: string },
  response: ReturnType<typeof acceptedHttpResponse>,
  error: IcnClientError,
): StreamFailure => Match.value(error).pipe(
  Match.tag("GeneratedClientRemoteError", (remote) => new StreamProviderError({
    call,
    response,
    providerError: {
      message: remote.body.error.message,
      type: remote.body.error.type,
      code: remote.body.error.code,
      param: null,
    },
    payload: payloadSample(JSON.stringify(remote.body)),
    progress: { dataPayloadsDecoded: 0, modelEventsEmitted: 0 },
  })),
  Match.tag("GeneratedClientTransportError", (transport) => new StreamOperationalFailure({
    call,
    response,
    progress: { dataPayloadsDecoded: 0, modelEventsEmitted: 0 },
    reason: {
      _tag: "BodyReadFailure",
      readError: {
        _tag: "EffectResponseBodyError",
        effectReason: "ICN inference stream failed",
        cause: toCauseInfo(transport.cause),
      },
    },
  })),
  Match.tag("GeneratedClientInvalidResponseError", (invalid) => new StreamProviderCorrectnessViolation({
    call,
    response,
    progress: { dataPayloadsDecoded: 0, modelEventsEmitted: 0 },
    violation: {
      _tag: "InvalidProviderChunk",
      problem: {
        _tag: "InvalidChunkSchema",
        payload: payloadSample(""),
        issue: { message: invalid.message },
        cause: Option.match(invalid.cause, { onNone: () => toCauseInfo(invalid), onSome: toCauseInfo }),
      },
    },
  })),
  Match.tag("GeneratedClientIncompleteStreamError", () => new StreamOperationalFailure({
    call,
    response,
    progress: { dataPayloadsDecoded: 0, modelEventsEmitted: 0 },
    reason: {
      _tag: "ConnectionClosedWithoutTerminalOutcome",
      expectation: { _tag: "FinishReasonOrMoreChunks" },
    },
  })),
  Match.tag("GeneratedClientInputError", (input) => new StreamClientCorrectnessViolation({
    call,
    response,
    component: "transport",
    message: `Generated ICN stream input was invalid at ${input.location}`,
    evidence: { _tag: "UnexpectedDefectCaught", cause: toCauseInfo(input.cause) },
    progress: { dataPayloadsDecoded: 0, modelEventsEmitted: 0 },
  })),
  Match.exhaustive,
)

const bindIcnModel = (
  client: IcnClientService,
  providerModelId: ProviderModelId,
  contextWindow: number,
  bindOptions?: ProviderModelBindOptions,
) => Effect.succeed({
  stream: (prompt: Prompt, tools: readonly ToolDefinition[], requestOptions?: BaseCallOptions & { generateToolCallId?: () => ToolCallId }) => {
    const call = {
      provider: "local",
      model: providerModelId,
      method: "POST" as const,
      url: `icn://chat/${encodeURIComponent(providerModelId)}`,
    }
    const defaults = bindOptions?.defaults
    const maxTokens = Option.fromNullable(requestOptions?.maxTokens ?? defaults?.maxTokens)
    const toolChoice = Option.fromNullable(requestOptions?.toolChoice ?? defaults?.toolChoice)
    const reasoningEffort = Option.fromNullable(requestOptions?.reasoningEffort ?? defaults?.reasoningEffort)
    const wire = {
      stream: true as const,
      stream_options: { include_usage: true },
      ...nativeChatCompletionsCodec.encodePrompt(providerModelId, prompt, tools),
      ...Option.match(maxTokens, { onNone: () => ({}), onSome: (max_tokens) => ({ max_tokens }) }),
      ...Option.match(toolChoice, { onNone: () => ({}), onSome: (tool_choice) => ({ tool_choice }) }),
      ...Option.match(reasoningEffort, { onNone: () => ({}), onSome: (reasoning_effort) => ({ reasoning_effort }) }),
    }
    return Schema.decodeUnknown(Generated.ChatCompletionRequest)(wire).pipe(
      Effect.mapError((cause) => new StreamStartClientCorrectnessViolation({
        call,
        component: "request_builder",
        message: "Unable to encode the ICN chat request",
        evidence: { _tag: "RequestBodyEncodingFailed", cause: toCauseInfo(cause) },
      })),
      Effect.tap(() => client.models.configureModelServing({
        path: { model_id: providerModelId },
        payload: { context_length: contextWindow, parallel_sequences: 1 },
      }).pipe(Effect.mapError((cause) => generatedStartFailure(call, cause)))),
      Effect.tap(() => bindOptions?.requestAttribution?.requestStarted ?? Effect.void),
      Effect.flatMap((payload) =>
        client.chat.createChatCompletion({ payload }).pipe(
            Effect.mapError((cause) => generatedStartFailure(call, cause)),
            Effect.map(({ status, headers, events }) => {
              const response = acceptedHttpResponse(status, headers)
              const chunks = events.pipe(
                Stream.mapEffect((chunk) => Schema.encode(Generated.ChatCompletionChunk)(chunk).pipe(
                  Effect.flatMap(Schema.decodeUnknown(ChatCompletionsStreamChunk)),
                  Effect.mapError((cause) => new GeneratedClientInvalidResponseError({
                    operationId: "createChatCompletion",
                    status,
                    message: "ICN chat chunk did not match the provider-neutral chat schema",
                    cause: Option.some(cause),
                  })),
                )),
              )
              const decoded = nativeChatCompletionsCodec.decode(chunks, {
                tools,
                streamContext: { call, response, responseHeaders: new Headers(headers) },
                ...(requestOptions?.generateToolCallId ? { generateToolCallId: requestOptions.generateToolCallId } : {}),
                toStreamFailure: (cause) => generatedBodyFailure(call, response, cause),
              })
              return {
                ...decoded,
                events: decoded.events,
                requestId: response.requestId,
              }
            }),
        )),
    )
  },
})

const bindMissingIcnModel = (providerModelId: ProviderModelId) => Effect.succeed({
  stream: () => Effect.fail(new StreamStartClientCorrectnessViolation({
    call: {
      provider: "local",
      model: providerModelId,
      method: "POST",
      url: `icn://chat/${encodeURIComponent(providerModelId)}`,
    },
    component: "request_builder",
    message: `No native ICN identity is associated with local provider model ${providerModelId}`,
    evidence: {
      _tag: "UnexpectedDefectCaught",
      cause: {
        _tag: "ErrorCause",
        name: "LocalModelIdentityNotFound",
        message: `No native ICN identity is associated with ${providerModelId}`,
      },
    },
  })),
})

export interface IcnProviderService extends LocalProviderSource {}

export class IcnProvider extends Context.Tag("@magnitudedev/icn/IcnProvider")<
  IcnProvider,
  IcnProviderService
>() {}

export const makeIcnProvider = (): Layer.Layer<
  IcnProvider,
  never,
  IcnClient | IcnInventory | IcnRecipes
> => Layer.effect(
  IcnProvider,
  Effect.gen(function* () {
    const client = yield* IcnClient
    const inventory = yield* IcnInventory
    const recipes = yield* IcnRecipes
    const associations = Effect.all({ inventory: inventory.get, recipes: recipes.get }).pipe(Effect.map(({ inventory, recipes }) => {
      const recommendations = recipes.state._tag === "Ready" ? recipes.state.recommendations : []
      const recommendationByNativeId = new Map(recommendations.flatMap((recommendation) => Option.match(
        recommendation.modelId,
        { onNone: () => [], onSome: (modelId) => [[modelId, recommendation] as const] },
      )))
      return inventory.state.data.flatMap((model) => {
        const recommendation = recommendationByNativeId.get(NativeIcnModelIdSchema.make(model.id))
        const contextWindow = recommendation?.contextWindow ?? (model.properties.type === "inspected"
          ? Option.getOrUndefined(Option.filter(
              Option.flatMap(model.properties.training_context_length, Option.fromNullable),
              (value) => value > 0,
            ))
          : undefined)
        if (contextWindow === undefined) return []
        const localModelId = recommendation
          ? candidateLocalModelId(recommendation)
          : nativeLocalModelId(NativeIcnModelIdSchema.make(model.id))
        const publicId = localProviderModelId(localModelId)
        return [{ model, publicId, contextWindow }]
      })
    }))
    const list = Effect.succeed([])
    const refresh = inventory.refresh.pipe(
      Effect.mapError((cause) => catalogError("Unable to refresh local models from ICN", cause)),
      Effect.zipRight(list),
    )
    const catalog = {
      list,
      refresh,
      get: (providerId: ProviderId, providerModelId: ProviderModelId) => Effect.fail(
        catalogError(`Local provider catalog is product-owned; cannot look up ${providerId}/${providerModelId}`),
      ),
    }
    return IcnProvider.of({
      catalog,
      bindModel: (providerModelId, options) => associations.pipe(
        Effect.flatMap((values) => {
          const association = values.find((value) => value.publicId === providerModelId)
          return association
            ? bindIcnModel(
                client,
                ProviderModelIdSchema.make(association.model.id),
                association.contextWindow,
                options,
              )
            : bindMissingIcnModel(providerModelId)
        }),
      ),
      discoverModelProperties: () => Effect.succeed(ModelDiscoveryOperationIdSchema.make("icn-authoritative")),
      status: client.system.health({}).pipe(
        Effect.mapError(generatedClientErrorMessage),
        Effect.match({
          onFailure: (message) => ({ status: "error" as const, message }),
          onSuccess: (health) => health.ready
            ? { status: "ok" as const }
            : { status: "loading" as const, message: health.status },
        }),
      ),
    })
  }),
)
