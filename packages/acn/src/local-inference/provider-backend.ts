import { Context, Effect, Layer } from "effect"
import {
  LlamaCppProviderBackendError,
  ModelCatalogError,
  type LlamaCppProviderBackend,
} from "@magnitudedev/sdk"
import {
  LlamaCppHost,
  LlamaCppModelStore,
  LlamaCppRuntime,
} from "@magnitudedev/llamacpp"
import type { DurableLocalModelBinding } from "@magnitudedev/storage"
import { catalogEntryForModelReferences } from "./catalog"
import { LocalModelConfiguration } from "./model-configuration"
import { providerModelIdForArtifact } from "./identity"
import { estimateRuntimeOverheadPerSlot } from "./recommendations"

const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const

export class LocalModelProviderBackend extends Context.Tag("LocalModelProviderBackend")<
  LocalModelProviderBackend,
  LlamaCppProviderBackend
>() {}

const backendError = (
  modelId: string,
  reason: string,
  cause?: unknown,
): LlamaCppProviderBackendError => new LlamaCppProviderBackendError({
  operation: "resolve_connection",
  modelId,
  reason,
  ...(cause === undefined ? {} : { cause }),
})

export const LocalModelProviderBackendLive: Layer.Layer<
  LocalModelProviderBackend,
  never,
  | LlamaCppHost
  | LlamaCppModelStore
  | LlamaCppRuntime
  | LocalModelConfiguration
> = Layer.effect(
  LocalModelProviderBackend,
  Effect.gen(function* () {
    const context = yield* Effect.context<
      LlamaCppHost | LlamaCppModelStore | LlamaCppRuntime | LocalModelConfiguration
    >()
    const host = yield* LlamaCppHost
    const models = yield* LlamaCppModelStore
    const runtime = yield* LlamaCppRuntime
    const configuration = yield* LocalModelConfiguration

    const listModels = Effect.gen(function* () {
      const [snapshot, config] = yield* Effect.all([models.inspect, configuration.get], { concurrency: 2 })
      const binding = config.binding
      if (!binding) return []

      const artifact = snapshot.artifacts.find(
        (candidate) => providerModelIdForArtifact(candidate.modelId) === binding.providerModelId,
      )
      if (artifact) {
        return [{
          providerId: "llamacpp" as const,
          providerModelId: binding.providerModelId,
          displayName: artifact.metadata.displayName,
          modelFamilyId: "unknown",
          contextWindow: binding.contextTokens,
          maxOutputTokens: Math.min(binding.contextTokens, 8192),
          capabilities: { vision: artifact.hasVisionProjector },
          pricing: ZERO_PRICING,
          reasoningEfforts: ["none"],
          ...(artifact.metadata.architecture ? { modelArchitecture: artifact.metadata.architecture } : {}),
          ...(artifact.metadata.tokenizerModel ? { tokenizerModel: artifact.metadata.tokenizerModel } : {}),
          ...(artifact.metadata.tokenizerPre ? { tokenizerPre: artifact.metadata.tokenizerPre } : {}),
          ...(artifact.metadata.baseModelNames.length > 0 ? { baseModelNames: artifact.metadata.baseModelNames } : {}),
          serverContextSize: binding.contextTokens,
        }]
      }

      const catalogEntry = catalogEntryForModelReferences([binding.providerModelId])
      const observedDisplayName = catalogEntry
        ? null
        : yield* runtime.inspect.pipe(
          Effect.map((runtimeSnapshot) => {
            const servers = [
              ...(runtimeSnapshot.managed ? [runtimeSnapshot.managed] : []),
              ...runtimeSnapshot.external,
            ]
            return servers
              .flatMap((server) => server.models)
              .find((model) => model.providerModelId === binding.providerModelId)
              ?.displayName ?? null
          }),
          Effect.catchAll(() => Effect.succeed(null)),
        )

      return [{
        providerId: "llamacpp" as const,
        providerModelId: binding.providerModelId,
        displayName: catalogEntry?.displayName ?? observedDisplayName ?? binding.providerModelId,
        modelFamilyId: "unknown",
        contextWindow: binding.contextTokens,
        maxOutputTokens: Math.min(binding.contextTokens, 8192),
        capabilities: { vision: false },
        pricing: ZERO_PRICING,
        reasoningEfforts: ["none"],
        serverContextSize: binding.contextTokens,
      }]
    }).pipe(
      Effect.mapError((cause) => new ModelCatalogError({
        message: cause instanceof Error ? cause.message : String(cause),
        cause,
      })),
    )

    const resolveManaged = (
      providerModelId: string,
      binding: Extract<DurableLocalModelBinding, { _tag: "Managed" }>,
    ) => Effect.gen(function* () {
      const artifact = yield* models.resolve(binding.artifactId)
      const fitPlan = yield* host.plan({
        modelBytes: artifact.sizeBytes,
        contextBytesPerSlot: estimateRuntimeOverheadPerSlot(
          artifact.sizeBytes,
          binding.contextTokens,
          binding.parallelSlots,
        ),
        parallelSlots: binding.parallelSlots,
        modelLayerCount: artifact.metadata.layerCount,
      })
      if (!fitPlan.fits) return yield* backendError(providerModelId, "The desired local model no longer fits stable host capacity.")
      const target = yield* runtime.ensureServing({
        _tag: "Managed",
        modelId: binding.artifactId,
        providerModelId,
        contextTokens: binding.contextTokens,
        fitPlan,
      })
      return target.connection
    }).pipe(
      Effect.mapError((cause) => cause instanceof LlamaCppProviderBackendError
        ? cause
        : backendError(providerModelId, cause instanceof Error ? cause.message : String(cause), cause)),
    )

    const resolveExternal = (providerModelId: string, endpointConfigId: string, contextTokens: number) =>
      runtime.ensureServing({
        _tag: "External",
        connectionId: endpointConfigId,
        providerModelId,
        contextTokens,
      }).pipe(
        Effect.map((target) => target.connection),
        Effect.mapError((cause) => cause instanceof LlamaCppProviderBackendError
          ? cause
          : backendError(providerModelId, cause instanceof Error ? cause.message : String(cause), cause)),
      )

    const backend: LlamaCppProviderBackend = {
      listModels: listModels.pipe(
        Effect.provide(context),
      ),

      resolveConnection: (providerModelId) => configuration.get.pipe(
        Effect.flatMap((config) => {
          const binding = config.binding
          if (!binding || binding.providerModelId !== providerModelId) {
            return Effect.fail(backendError(providerModelId, "This model is not the active local binding."))
          }
          return binding._tag === "Managed"
            ? resolveManaged(providerModelId, binding)
            : resolveExternal(providerModelId, binding.endpointConfigId, binding.contextTokens)
        }),
        Effect.mapError((cause) => cause instanceof LlamaCppProviderBackendError
          ? cause
          : backendError(providerModelId, cause instanceof Error ? cause.message : String(cause), cause)),
        Effect.provide(context),
      ),

      status: Effect.gen(function* () {
        const config = yield* configuration.get
        if (!config.binding) {
          return { status: "not_found", message: "No local model is active." } as const
        }
        const snapshot = yield* runtime.inspect
        const observations = [...(snapshot.managed ? [snapshot.managed] : []), ...snapshot.external]
        const matching = observations.find((server) => server.models.some(
          (model) => model.providerModelId === config.binding?.providerModelId,
        ))
        if (!matching) return { status: "not_found", message: "The active local model is not currently serving." } as const
        return matching.health === "ready"
          ? { status: "ok", endpointLabel: matching.ownership } as const
          : matching.health === "loading"
            ? { status: "loading", endpointLabel: matching.ownership } as const
            : { status: "error", endpointLabel: matching.ownership, message: "The local runtime is unhealthy." } as const
      }).pipe(
        Effect.catchAll((cause) => Effect.succeed({
          status: "error",
          message: cause instanceof Error ? cause.message : String(cause),
        } as const)),
        Effect.provide(context),
      ),
    }
    return LocalModelProviderBackend.of(backend)
  }),
)
