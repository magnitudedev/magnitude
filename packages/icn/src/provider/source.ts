import { Array as Arr, Context, Effect, Layer, Match, Option, Predicate, Schema, Stream } from "effect"
import {
  AVAILABLE_PROVIDER_MODEL,
  ModelCatalogError,
  ModelDiscoveryOperationIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  StreamClientCorrectnessViolation,
  StreamOperationalFailure,
  StreamProviderCorrectnessViolation,
  StreamProviderError,
  StreamStartClientCorrectnessViolation,
  StreamStartOperationalFailure,
  StreamStartProviderCorrectnessViolation,
  VisionProperty,
  acceptedHttpResponse,
  payloadSample,
  rejectedHttpResponse,
  streamStartFailureFromRejectedResponse,
  toCauseInfo,
  nativeChatCompletionsCodec,
  type BaseCallOptions,
  type ChatCompletionsStreamChunk,
  type ProviderModelBindOptions,
  type ProviderId,
  type ProviderModelId,
  type StreamFailure,
  type StreamStartFailure,
  type Prompt,
  type ToolCallId,
  type ToolDefinition,
} from "@magnitudedev/ai"
import { IcnApiClient } from "../generated/client.js"
import * as Generated from "../generated/schemas.js"
import type { GeneratedClientError } from "@magnitudedev/openapi-effect/client-runtime"
import { IcnInventory, type IcnInventoryService } from "../inventory/index.js"
import { LocalModelInfoSchema, LocalProviderId, type LocalModelInfo } from "./contract.js"
import type { LocalProviderSource } from "./provider.js"

const PROVIDER_ID = LocalProviderId.make("local")
const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const

const catalogError = (message: string, cause?: unknown) => new ModelCatalogError({ message, cause })

const isProviderReadyStatus = (availability: Generated.ModelAvailabilitySchema): boolean =>
  availability.type === "available"

const reasoningFor = (properties: Generated.InventoryPropertiesSchema) => {
  if (properties.type !== "inspected" || properties.reasoning.type !== "supported") {
    const effort = ReasoningEffortSchema.make("none")
    return { defaultEffort: effort, property: new ReasoningProperty.states.Resolved({ value: [effort] }) }
  }
  const control = properties.reasoning.control
  if (control.type === "effort" && control.levels.length > 0) {
    const efforts = control.levels.map((level) => ReasoningEffortSchema.make(level))
    const requested = Option.filter(
      Option.map(Option.filter(control.default, Predicate.isNotNull), ReasoningEffortSchema.make),
      (effort) => efforts.includes(effort),
    )
    const fallback = Option.getOrElse(
      Arr.head(efforts),
      () => ReasoningEffortSchema.make("none"),
    )
    const defaultEffort = Option.getOrElse(requested, () => fallback)
    return { defaultEffort, property: new ReasoningProperty.states.Resolved({ value: efforts }) }
  }
  const efforts = ["none", "medium"].map((level) => ReasoningEffortSchema.make(level))
  const defaultEffort = Option.getOrElse(
    Arr.get(efforts, control.type === "toggle" && !control.default ? 0 : 1),
    () => ReasoningEffortSchema.make("none"),
  )
  return { defaultEffort, property: new ReasoningProperty.states.Resolved({ value: efforts }) }
}

export const icnModelToProviderModel = (model: Generated.Model): LocalModelInfo => {
  const inspected = model.properties.type === "inspected"
    ? Option.some(model.properties)
    : Option.none<Generated.InventoryPropertiesSchema & { readonly type: "inspected" }>()
  const reasoning = reasoningFor(model.properties)
  const serving = Option.filter(model.serving_configuration, Predicate.isNotNull)
  const contextWindow = Option.getOrElse(
    Option.orElse(
      Option.map(serving, (configuration) => configuration.profile.context_length),
      () => Option.flatMap(inspected, (properties) =>
        Option.filter(properties.training_context_length, Predicate.isNotNull)),
    ),
    () => 131_072,
  )
  const displayName = Option.getOrElse(
    Option.filter(
      Option.map(Option.filter(model.name, Predicate.isNotNull), (name) => name.trim()),
      (name) => name.length > 0,
    ),
    () => model.id,
  )
  return LocalModelInfoSchema.make({
    providerId: PROVIDER_ID,
    providerModelId: ProviderModelIdSchema.make(model.id),
    displayName,
    contextWindow: Math.max(1, contextWindow),
    maxOutputTokens: Math.max(1, Math.min(32_768, contextWindow)),
    defaultReasoningEffort: reasoning.defaultEffort,
    properties: {
      vision: new VisionProperty.states.Resolved({
        value: Option.exists(inspected, (properties) => properties.modalities.includes("vision")),
      }),
      reasoning: reasoning.property,
    },
    availability: Option.isSome(serving) && isProviderReadyStatus(model.availability) && model.hardware.type === "fits"
      ? AVAILABLE_PROVIDER_MODEL
      : { _tag: "Disabled", reason: model.hardware.type === "does_not_fit" ? "insufficient_resources" : "model_unavailable" },
    pricing: ZERO_PRICING,
  })
}

type IcnClientError = GeneratedClientError<Generated.ErrorResponse>

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
  client: IcnApiClient,
  inventory: IcnInventoryService,
  providerModelId: ProviderModelId,
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
      Effect.tap(() => bindOptions?.requestAttribution?.requestStarted ?? Effect.void),
      Effect.flatMap((payload) => inventory.observeChatAdmission(
        client.chat.createChatCompletion({ payload }).pipe(
            Effect.mapError((cause) => generatedStartFailure(call, cause)),
            Effect.map(({ status, headers, events }) => {
              const response = acceptedHttpResponse(status, headers)
              const chunks = events.pipe(
                Stream.ensuring(inventory.refresh.pipe(Effect.ignore)),
                Stream.map((chunk) => Schema.encodeSync(Generated.ChatCompletionChunk)(chunk) as ChatCompletionsStreamChunk),
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
        ),
      )),
    )
  },
})

export interface IcnProviderService extends LocalProviderSource {}

export class IcnProvider extends Context.Tag("@magnitudedev/icn/IcnProvider")<
  IcnProvider,
  IcnProviderService
>() {}

export const makeIcnProvider = (): Layer.Layer<
  IcnProvider,
  never,
  IcnApiClient | IcnInventory
> => Layer.effect(
  IcnProvider,
  Effect.gen(function* () {
    const client = yield* IcnApiClient
    const inventory = yield* IcnInventory
    const list = inventory.get.pipe(
      Effect.map(({ state }) => state.data.map(icnModelToProviderModel)),
    )
    const refresh = inventory.refresh.pipe(
      Effect.mapError((cause) => catalogError("Unable to refresh local models from ICN", cause)),
      Effect.zipRight(list),
    )
    const catalog = {
      list,
      refresh,
      get: (providerId: ProviderId, providerModelId: ProviderModelId) =>
        providerId !== PROVIDER_ID
          ? Effect.fail(catalogError(`Unknown local provider ${providerId}`))
          : list.pipe(Effect.flatMap((models) => {
            const model = models.find((candidate) => candidate.providerModelId === providerModelId)
            return model
              ? Effect.succeed(model)
              : Effect.fail(catalogError(`Unknown local model ${providerModelId}`))
          })),
    }
    return IcnProvider.of({
      catalog,
      bindModel: (providerModelId, options) => bindIcnModel(client, inventory, providerModelId, options),
      discoverModelProperties: () => Effect.succeed(ModelDiscoveryOperationIdSchema.make("icn-authoritative")),
      status: client.system.health({}).pipe(
        Effect.map((health) => health.ready
          ? { status: "ok" as const }
          : { status: "loading" as const, message: health.status }),
        Effect.catchAll((cause) => Effect.succeed({ status: "error" as const, message: String(cause) })),
      ),
    })
  }),
)
