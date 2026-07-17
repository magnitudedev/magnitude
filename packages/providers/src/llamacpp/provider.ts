import { Data, Effect, Exit, Option, Redacted, Scope, Stream } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import type {
  BaseCallOptions,
  BoundModel,
  ModelCatalog,
  Provider,
  ProviderModelBindOptions,
  ModelPropertyDiscoveryRequest,
  ModelDiscoveryOperationId,
  ModelPropertyDiscoveryError,
} from "@magnitudedev/ai"
import { ProviderIdSchema, StreamStartOperationalFailure, type ProviderModelId, type ReasoningEffort } from "@magnitudedev/ai"
import { createLlamaCppCompatibleSpec, wrapAsBaseModel } from "./models"
import type { LlamaCppCallOptions, LlamaCppModelInfo, LlamaServedModelId, LlamaServingRouteId } from "./contract"

export const PROVIDER_ID = ProviderIdSchema.make("llamacpp")

export class LlamaCppAcquisitionError extends Data.TaggedError("LlamaCppAcquisitionError")<{
  readonly modelId: ProviderModelId
  readonly reason: string
  readonly cause?: unknown
}> {}

export interface LlamaCppInferenceLease {
  readonly providerModelId: ProviderModelId
  readonly routeId: LlamaServingRouteId
  readonly origin: URL
  readonly authorization: Option.Option<Redacted.Redacted<string>>
  readonly servedModelId: LlamaServedModelId
  readonly reasoningEffort: ReasoningEffort
  readonly chatTemplateKwargs: Option.Option<Readonly<Record<string, unknown>>>
  readonly thinkingBudgetTokens: Option.Option<number>
}

/**
 * ACN-owned product projection consumed by the protocol-only provider.
 * Acquisition is scoped: the caller must keep Scope alive for the returned
 * model stream and closing it releases the instance lease.
 */
export interface LlamaCppProviderSource {
  readonly catalog: ModelCatalog<LlamaCppModelInfo>
  readonly discoverModelProperties: (
    request: ModelPropertyDiscoveryRequest,
  ) => Effect.Effect<ModelDiscoveryOperationId, ModelPropertyDiscoveryError>
  readonly acquire: (
    providerModelId: ProviderModelId,
    reasoningEffort: ReasoningEffort | undefined,
  ) => Effect.Effect<LlamaCppInferenceLease, LlamaCppAcquisitionError, Scope.Scope>
  readonly status: Effect.Effect<{
    readonly status: "ok" | "loading" | "not_found" | "error"
    readonly message?: string
    readonly hint?: string
  }, never, HttpClient.HttpClient>
}

export interface LlamaCppProviderInstance {
  readonly provider: Provider<LlamaCppModelInfo>
  readonly checkStatus: LlamaCppProviderSource["status"]
}

const callOptions = (defaults: Partial<BaseCallOptions> | undefined): Partial<LlamaCppCallOptions> | undefined =>
  defaults
    ? {
        ...(defaults.maxTokens === undefined ? {} : { maxTokens: defaults.maxTokens }),
        ...(defaults.toolChoice === undefined ? {} : { toolChoice: defaults.toolChoice }),
      }
    : undefined

const authFor = (lease: LlamaCppInferenceLease) => (headers: Headers): void => {
  if (Option.isSome(lease.authorization)) {
    headers.set("Authorization", `Bearer ${Redacted.value(lease.authorization.value)}`)
  }
}

const inferenceEndpoint = (lease: LlamaCppInferenceLease): string =>
  `${lease.origin.toString().replace(/\/+$/, "")}/v1`

const dynamicBoundModel = (
  source: LlamaCppProviderSource,
  providerModelId: ProviderModelId,
  options: ProviderModelBindOptions | undefined,
): BoundModel<BaseCallOptions> => ({
  stream: (prompt, tools, requestOptions) => Effect.gen(function* () {
    const scope = yield* Scope.make()
    return yield* Effect.gen(function* () {
      const reasoningEffort = requestOptions?.reasoningEffort ?? options?.defaults?.reasoningEffort
      const lease = yield* source.acquire(providerModelId, reasoningEffort).pipe(
        Effect.provideService(Scope.Scope, scope),
        Effect.mapError((cause) => new StreamStartOperationalFailure({
          call: {
            provider: PROVIDER_ID,
            model: providerModelId,
            method: "POST",
            url: `llamacpp://acquire/${encodeURIComponent(providerModelId)}`,
          },
          reason: {
            _tag: "RequestFailedBeforeResponse",
            cause: { _tag: "ErrorCause", name: cause._tag, message: cause.reason },
          },
        })),
      )
      if (requestOptions?.reasoningEffort === undefined
        && reasoningEffort !== undefined
        && lease.reasoningEffort !== reasoningEffort) {
        yield* (options?.reasoningEffortFallback?.(reasoningEffort, lease.reasoningEffort) ?? Effect.void).pipe(
          Effect.mapError((cause) => new StreamStartOperationalFailure({
            call: {
              provider: PROVIDER_ID,
              model: providerModelId,
              method: "POST",
              url: `llamacpp://reasoning-fallback/${encodeURIComponent(providerModelId)}`,
            },
            reason: {
              _tag: "RequestFailedBeforeResponse",
              cause: { _tag: "ErrorCause", name: "ReasoningEffortFallbackFailed", message: cause instanceof Error ? cause.message : String(cause) },
            },
          })),
        )
      }
      const result = yield* wrapAsBaseModel(
        createLlamaCppCompatibleSpec({
          modelId: lease.servedModelId,
          endpoint: inferenceEndpoint(lease),
        }).bind({
          auth: authFor(lease),
          defaults: {
            ...callOptions(options?.defaults),
            ...(Option.isSome(lease.chatTemplateKwargs) ? { chatTemplateKwargs: lease.chatTemplateKwargs.value } : {}),
            ...(Option.isSome(lease.thinkingBudgetTokens) ? { thinkingBudgetTokens: lease.thinkingBudgetTokens.value } : {}),
          },
          ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
        }),
      ).stream(prompt, tools, requestOptions)
      yield* (options?.requestAttribution?.requestStarted ?? Effect.void)
      return {
        ...result,
        events: result.events.pipe(Stream.ensuring(Scope.close(scope, Exit.void))),
      }
    }).pipe(Effect.onError(() => Scope.close(scope, Exit.void)))
  }),
})

export function createLlamaCppProvider(source: LlamaCppProviderSource): LlamaCppProviderInstance {
  const provider: Provider<LlamaCppModelInfo> = {
    id: PROVIDER_ID,
    displayName: "Llama.cpp",
    catalog: source.catalog,
    discoverModelProperties: source.discoverModelProperties,
    bindModel: (providerModelId, options) => Effect.succeed(dynamicBoundModel(source, providerModelId, options)),
    classifyModelFamily: () => Option.none(),
  }
  return { provider, checkStatus: source.status }
}
