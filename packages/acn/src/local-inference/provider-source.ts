import { Context, Effect, Exit, Layer, Match, Option, Ref, Schema, Stream } from "effect"
import {
  AVAILABLE_PROVIDER_MODEL,
  LocalModelInfoSchema,
  LocalProviderId,
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
  type LocalModelInfo,
  type LocalProviderSource,
  type ProviderModelBindOptions,
  type ProviderId,
  type ProviderModelId,
  type StreamFailure,
  type StreamStartFailure,
  type Prompt,
  type ToolCallId,
  type ToolDefinition,
} from "@magnitudedev/sdk"
import { IcnApiClient, Generated, type GeneratedClientError } from "@magnitudedev/icn"
import { LocalModelInventoryChanges } from "./inventory-changes"
import { LocalModelConfiguration } from "./model-configuration"
import { reconcileSelectedServingConfiguration } from "./serving-configuration"
import { AcnActivityTracker, type AcnActivityTrackerApi } from "../activity-tracker"

const PROVIDER_ID = LocalProviderId.make("local")
const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const

const catalogError = (message: string, cause?: unknown) => new ModelCatalogError({ message, cause })

const optionValue = <A>(value: Option.Option<A>): A | undefined => Option.getOrUndefined(value)

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
    const requested = Option.getOrUndefined(control.default)
    const defaultEffort = requested && efforts.includes(ReasoningEffortSchema.make(requested))
      ? ReasoningEffortSchema.make(requested)
      : efforts[0]!
    return { defaultEffort, property: new ReasoningProperty.states.Resolved({ value: efforts }) }
  }
  const efforts = ["none", "medium"].map((level) => ReasoningEffortSchema.make(level))
  const defaultEffort = control.type === "toggle" && !control.default ? efforts[0]! : efforts[1]!
  return { defaultEffort, property: new ReasoningProperty.states.Resolved({ value: efforts }) }
}

export const icnModelToProviderModel = (model: Generated.Model): LocalModelInfo => {
  const inspected = model.properties.type === "inspected" ? model.properties : undefined
  const reasoning = reasoningFor(model.properties)
  const serving = optionValue(model.serving_configuration)
  const contextWindow = serving?.profile.context_length
    ?? (inspected ? optionValue(inspected.training_context_length) : undefined)
    ?? 131_072
  const displayName = Option.getOrUndefined(model.name)?.trim() || model.id
  return LocalModelInfoSchema.make({
    providerId: PROVIDER_ID,
    providerModelId: ProviderModelIdSchema.make(model.id),
    displayName,
    contextWindow: Math.max(1, contextWindow),
    maxOutputTokens: Math.max(1, Math.min(32_768, contextWindow)),
    defaultReasoningEffort: reasoning.defaultEffort,
    properties: {
      vision: new VisionProperty.states.Resolved({ value: inspected?.modalities.includes("vision") ?? false }),
      reasoning: reasoning.property,
    },
    availability: serving && isProviderReadyStatus(model.availability) && model.hardware.type === "fits"
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
  activity: AcnActivityTrackerApi,
  onRuntimeChange: Effect.Effect<void>,
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
    const wire = {
      stream: true as const,
      stream_options: { include_usage: true },
      ...nativeChatCompletionsCodec.encodePrompt(providerModelId, prompt, tools),
      ...((requestOptions?.maxTokens ?? defaults?.maxTokens) === undefined
        ? {}
        : { max_tokens: requestOptions?.maxTokens ?? defaults?.maxTokens }),
      ...((requestOptions?.toolChoice ?? defaults?.toolChoice) === undefined
        ? {}
        : { tool_choice: requestOptions?.toolChoice ?? defaults?.toolChoice }),
      ...((requestOptions?.reasoningEffort ?? defaults?.reasoningEffort) === undefined
        ? {}
        : { reasoning_effort: requestOptions?.reasoningEffort ?? defaults?.reasoningEffort }),
    }
    return Schema.decodeUnknown(Generated.ChatCompletionRequest)(wire).pipe(
      Effect.mapError((cause) => new StreamStartClientCorrectnessViolation({
        call,
        component: "request_builder",
        message: "Unable to encode the ICN chat request",
        evidence: { _tag: "RequestBodyEncodingFailed", cause: toCauseInfo(cause) },
      })),
      Effect.tap(() => bindOptions?.requestAttribution?.requestStarted ?? Effect.void),
      Effect.flatMap((payload) => Effect.uninterruptibleMask((restore) =>
        activity.acquireActiveWork(`local-inference:chat:${providerModelId}`).pipe(
          Effect.flatMap((releaseActiveWork) => restore(client.chat.createChatCompletion({ payload })).pipe(
            Effect.mapError((cause) => generatedStartFailure(call, cause)),
            Effect.tap(() => onRuntimeChange),
            Effect.map(({ status, headers, events }) => {
              const response = acceptedHttpResponse(status, headers)
              const chunks = events.pipe(
                Stream.ensuring(onRuntimeChange),
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
                events: decoded.events.pipe(Stream.ensuring(releaseActiveWork)),
                requestId: response.requestId,
              }
            }),
            Effect.onExit((exit) => Exit.isSuccess(exit) ? Effect.void : releaseActiveWork),
          )),
        ),
      )),
    )
  },
})

export interface LocalModelProviderSourceApi extends LocalProviderSource {
  readonly changes: Stream.Stream<void>
}

export class LocalModelProviderSource extends Context.Tag("LocalModelProviderSource")<
  LocalModelProviderSource,
  LocalModelProviderSourceApi
>() {}

export const LocalModelProviderSourceLive: Layer.Layer<
  LocalModelProviderSource,
  never,
  IcnApiClient | LocalModelInventoryChanges | LocalModelConfiguration | AcnActivityTracker
> = Layer.effect(
  LocalModelProviderSource,
  Effect.gen(function* () {
    const client = yield* IcnApiClient
    const changes = yield* LocalModelInventoryChanges
    const configuration = yield* LocalModelConfiguration
    const activity = yield* AcnActivityTracker
    type CachedCatalog = { readonly revision: number; readonly models: readonly LocalModelInfo[] }
    const cache = yield* Ref.make<CachedCatalog | null>(null)
    const lock = yield* Effect.makeSemaphore(1)
    const fetch = client.models.listModels({}).pipe(
      Effect.flatMap(({ data }) => reconcileSelectedServingConfiguration(client, configuration, data)),
      Effect.map((models) => models.map(icnModelToProviderModel)),
      Effect.mapError((cause) => catalogError("Unable to list local models from ICN", cause)),
    )
    const fetchStable: Effect.Effect<readonly LocalModelInfo[], ModelCatalogError> = Effect.gen(function* () {
      const before = yield* changes.revision
      const models = yield* fetch
      const after = yield* changes.revision
      if (before !== after) return yield* Effect.suspend(() => fetchStable)
      yield* Ref.set(cache, { revision: after, models })
      return models
    })
    const refresh = lock.withPermits(1)(fetchStable)
    const list = Effect.gen(function* () {
      const revision = yield* changes.revision
      const cached = yield* Ref.get(cache)
      if (cached?.revision === revision) return cached.models
      return yield* lock.withPermits(1)(Effect.gen(function* () {
        const joinedRevision = yield* changes.revision
        const joined = yield* Ref.get(cache)
        return joined?.revision === joinedRevision ? joined.models : yield* fetchStable
      }))
    })
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
    return LocalModelProviderSource.of({
      catalog,
      bindModel: (providerModelId, options) => bindIcnModel(client, activity, changes.publish, providerModelId, options),
      discoverModelProperties: () => Effect.succeed(ModelDiscoveryOperationIdSchema.make("icn-authoritative")),
      status: client.system.health({}).pipe(
        Effect.map((health) => health.ready
          ? { status: "ok" as const }
          : { status: "loading" as const, message: health.status }),
        Effect.catchAll((cause) => Effect.succeed({ status: "error" as const, message: String(cause) })),
      ),
      changes: changes.stream,
    })
  }),
)
