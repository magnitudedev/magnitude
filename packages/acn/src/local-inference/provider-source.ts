import { Context, Data, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  AVAILABLE_PROVIDER_MODEL,
  LocalModelInfoSchema,
  LocalProviderId,
  ModelCatalogError,
  ModelDiscoveryOperationIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  StreamOperationalFailure,
  StreamStartClientCorrectnessViolation,
  StreamStartOperationalFailure,
  VisionProperty,
  acceptedHttpResponse,
  nativeChatCompletionsCodec,
  type BaseCallOptions,
  type ChatCompletionsStreamChunk,
  type LocalModelInfo,
  type LocalProviderSource,
  type ProviderModelBindOptions,
  type ProviderId,
  type ProviderModelId,
  type StreamFailure,
  type Prompt,
  type ToolCallId,
  type ToolDefinition,
} from "@magnitudedev/sdk"
import { IcnApiClient, Generated } from "@magnitudedev/icn"
import { LocalModelInventoryChanges } from "./inventory-changes"

const PROVIDER_ID = LocalProviderId.make("local")
const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const

class IcnModelNotFittable extends Data.TaggedError("IcnModelNotFittable")<{
  readonly modelId: string
  readonly message: string
}> {}

const catalogError = (message: string, cause?: unknown) => new ModelCatalogError({ message, cause })

const optionValue = <A>(value: Option.Option<A>): A | undefined => Option.getOrUndefined(value)

const isProviderReadyStatus = (status: Generated.ModelStatusSchema): boolean =>
  status.type === "available" || status.type === "loaded" || status.type === "load_failed"

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
  const contextWindow = inspected
    ? optionValue(inspected.training_context_length) ?? 131_072
    : 131_072
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
    availability: isProviderReadyStatus(model.status) && model.hardware.type === "fits"
      ? AVAILABLE_PROVIDER_MODEL
      : { _tag: "Disabled", reason: model.hardware.type === "does_not_fit" ? "insufficient_resources" : "model_unavailable" },
    pricing: ZERO_PRICING,
  })
}

const generatedFailure = (call: { provider: string; model: string; method: "POST"; url: string }, cause: unknown): StreamFailure =>
  new StreamOperationalFailure({
    call,
    response: acceptedHttpResponse(200, {}),
    progress: { dataPayloadsDecoded: 0, modelEventsEmitted: 0 },
    reason: {
      _tag: "BodyReadFailure",
      readError: {
        _tag: "EffectResponseBodyError",
        effectReason: "ICN inference stream failed",
        cause: {
          _tag: "ErrorCause",
          name: typeof cause === "object" && cause && "_tag" in cause ? String(cause._tag) : "IcnStreamError",
          message: cause instanceof Error ? cause.message : String(cause),
        },
      },
    },
  })

const bindIcnModel = (
  client: IcnApiClient,
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
        evidence: { _tag: "RequestBodyEncodingFailed", cause: { _tag: "ErrorCause", name: "ParseError", message: String(cause) } },
      })),
      Effect.tap(() => Effect.gen(function* () {
        const runtime = yield* client.runtime.getRuntimeState({})
        if (runtime.status.type === "ready" && runtime.status.model_id === providerModelId) return
        const model = yield* client.models.getModel({ path: { model_id: providerModelId } })
        if (model.hardware.type !== "fits") {
          return yield* new IcnModelNotFittable({
            modelId: providerModelId,
            message: `ICN has no fitting execution profile for ${providerModelId}`,
          })
        }
        const hardware = yield* client.system.getHardware({})
        yield* client.runtime.loadRuntimeModel({ payload: {
          model_id: providerModelId,
          profile: {
            policy: hardware.assessment_policy,
            context_length: model.hardware.profile.context_length,
            parallel_sequences: 1,
          },
        }}).pipe(Stream.runDrain)
      }).pipe(Effect.mapError((cause) => new StreamStartOperationalFailure({
        call,
        reason: {
          _tag: "RequestFailedBeforeResponse",
          cause: {
            _tag: "ErrorCause",
            name: typeof cause === "object" && cause && "_tag" in cause ? String(cause._tag) : "IcnRuntimeError",
            message: cause instanceof Error ? cause.message : String(cause),
          },
        },
      })))),
      Effect.map((payload) => {
        const chunks = client.chat.createChatCompletion({ payload }).pipe(
          Stream.map((chunk) => Schema.encodeSync(Generated.ChatCompletionChunk)(chunk) as ChatCompletionsStreamChunk),
        )
        const decoded = nativeChatCompletionsCodec.decode(chunks, {
          tools,
          streamContext: { call, response: acceptedHttpResponse(200, {}), responseHeaders: new Headers() },
          ...(requestOptions?.generateToolCallId ? { generateToolCallId: requestOptions.generateToolCallId } : {}),
          toStreamFailure: (cause) => generatedFailure(call, cause),
        })
        return { ...decoded, requestId: null }
      }),
      Effect.tap(() => bindOptions?.requestAttribution?.requestStarted ?? Effect.void),
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

export const LocalModelProviderSourceLive = Layer.effect(
  LocalModelProviderSource,
  Effect.gen(function* () {
    const client = yield* IcnApiClient
    const changes = yield* LocalModelInventoryChanges
    const list = client.models.listModels({}).pipe(
      Effect.map(({ data }) => data.map(icnModelToProviderModel)),
      Effect.mapError((cause) => catalogError("Unable to list local models from ICN", cause)),
    )
    const catalog = {
      list,
      refresh: list,
      get: (providerId: ProviderId, providerModelId: ProviderModelId) =>
        providerId !== PROVIDER_ID
          ? Effect.fail(catalogError(`Unknown local provider ${providerId}`))
          : client.models.getModel({ path: { model_id: providerModelId } }).pipe(
              Effect.map(icnModelToProviderModel),
              Effect.mapError((cause) => catalogError(`Unknown local model ${providerModelId}`, cause)),
            ),
    }
    return LocalModelProviderSource.of({
      catalog,
      bindModel: (providerModelId, options) => bindIcnModel(client, providerModelId, options),
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
