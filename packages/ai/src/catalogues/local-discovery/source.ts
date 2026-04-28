import { Effect, Option } from "effect"
import { CatalogueConfig, type ProviderOptions } from "../config"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import type { ProviderModel } from "../../lib/model/provider-model"
import type { CatalogueSource } from "../types"
import { discoverLlamaCppModels } from "./strategies/llamacpp"
import { discoverLmStudioModels, type LocalDiscoveryResult } from "./strategies/lmstudio"
import { discoverOpenAIModels } from "./strategies/openai-models"
import { discoverOllamaModels } from "./strategies/ollama"

const LOCAL_PROVIDER_IDS = new Set(["lmstudio", "ollama", "llama.cpp", "openai-compatible-local"])

function mergeDiscoveredAndRemembered(
  discoveredModels: readonly ProviderModel[],
  rememberedModelIds: readonly string[] | undefined,
  provider: ProviderDefinition,
): readonly ProviderModel[] {
  const byId = new Map<string, ProviderModel>()
  for (const model of discoveredModels) {
    byId.set(model.id, model)
  }

  for (const id of rememberedModelIds ?? []) {
    if (!id || byId.has(id)) {
      continue
    }

    const now = new Date().toISOString()
    byId.set(id, {
      id,
      providerId: provider.id,
      providerName: provider.name,
      canonicalModelId: null,
      name: id,
      contextWindow: 200_000,
      maxContextTokens: null,
      maxOutputTokens: null,
      supportsToolCalls: true,
      supportsReasoning: false,
      supportsVision: false,
      costs: {
        inputPerM: 0,
        outputPerM: 0,
        cacheReadPerM: null,
        cacheWritePerM: null,
      },
      releaseDate: now,
      discovery: {
        primarySource: "local",
        fetchedAt: now,
      },
    })
  }

  return [...byId.values()]
}

function discoverForProvider(
  provider: ProviderDefinition,
  baseUrl: string,
): Effect.Effect<LocalDiscoveryResult, never> {
  switch (provider.id) {
    case "lmstudio":
      return discoverLmStudioModels(provider.id, provider.name, baseUrl)
    case "ollama":
      return discoverOllamaModels(provider.id, provider.name, baseUrl)
    case "llama.cpp":
      return discoverLlamaCppModels(provider.id, provider.name, baseUrl)
    case "openai-compatible-local":
      return discoverOpenAIModels(provider.id, provider.name, baseUrl).pipe(
        Effect.map((models) => ({
          models,
          error: null,
          source: "openai-v1-models",
          status: models.length > 0 ? "success_non_empty" : "success_empty",
          diagnostics: [],
        }) as LocalDiscoveryResult),
        Effect.catchAll((error) =>
          Effect.succeed<LocalDiscoveryResult>({
            models: [],
            error: error.message || "discovery failed",
            source: null,
            status: "failure",
            diagnostics: [],
          }),
        ),
      )
    default:
      return Effect.succeed({
        models: [],
        error: null,
        source: null,
        status: "success_empty",
        diagnostics: [],
      })
  }
}

function persistDiscovery(
  providerId: string,
  discovery: LocalDiscoveryResult,
): Effect.Effect<void, never, CatalogueConfig> {
  const now = new Date().toISOString()

  return Effect.flatMap(CatalogueConfig, (config) =>
    config.setProviderOptions(providerId, (current) => ({
      ...(current ?? {}),
      discoveredModels: discovery.models.map((model) => ({
        id: model.id,
        name: model.name,
        maxContextTokens: model.maxContextTokens ?? null,
        discoveredAt: now,
        source: discovery.source ?? "unknown",
      })),
      inventoryUpdatedAt: now,
      lastDiscoveryStatus: discovery.status,
      lastDiscoverySource: discovery.source ?? undefined,
      lastDiscoveryDiagnostics:
        discovery.diagnostics.length > 0 ? [...discovery.diagnostics] : undefined,
      lastDiscoveryError: discovery.error ?? undefined,
    })),
  )
}

function getProviderOptions(
  providerId: string,
): Effect.Effect<ProviderOptions | undefined, never> {
  return Effect.gen(function* () {
    const config = yield* Effect.serviceOption(CatalogueConfig)
    return yield* Option.match(config, {
      onNone: () => Effect.as(Effect.void, undefined as ProviderOptions | undefined),
      onSome: (service) => service.getProviderOptions(providerId).pipe(Effect.orElseSucceed(() => undefined)),
    })
  })
}

export function makeLocalDiscoverySource(
  provider: ProviderDefinition,
): CatalogueSource | null {
  if (provider.family !== "local" || !LOCAL_PROVIDER_IDS.has(provider.id)) {
    return null
  }

  return {
    id: `local-discovery:${provider.id}`,
    fetch: Effect.gen(function* () {
      const options = yield* getProviderOptions(provider.id)
      const effectiveBaseUrl = options?.baseUrl?.trim() || provider.defaultBaseUrl

      if (!effectiveBaseUrl) {
        return mergeDiscoveredAndRemembered([], options?.rememberedModelIds, provider)
      }

      const discovery = yield* discoverForProvider(provider, effectiveBaseUrl)
      const merged = mergeDiscoveredAndRemembered(
        discovery.models,
        options?.rememberedModelIds,
        provider,
      )

      const config = yield* Effect.serviceOption(CatalogueConfig)
      if (Option.isSome(config)) {
        yield* persistDiscovery(provider.id, discovery).pipe(
          Effect.provideService(CatalogueConfig, config.value),
        )
      } else {
        yield* Effect.void
      }

      return merged
    }),
  }
}
