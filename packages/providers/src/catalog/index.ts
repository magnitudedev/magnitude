import { Effect, Layer, Option } from 'effect'
import {
  CatalogCache,
  MODELS_DEV_TTL_MS,
  OPENROUTER_TTL_MS,
} from '@magnitudedev/storage'
import type { CatalogCacheService } from '@magnitudedev/storage'

import { ModelCatalog } from './contracts'
import { mergeProviderModels } from './merge'
import { fetchModelsDevData, normalizeModelsDevProvider } from './models-dev-source'
import { fetchOpenRouterModels, normalizeOpenRouterModels } from './openrouter-source'
import { makeRefreshSchedule } from './refresh'
import { getProvider, getStaticProviderModels, PROVIDERS, setProviderModels } from '../registry'
import type { ModelDefinition } from '../types'
import { discoverLocalProviderModels, mergeDiscoveredAndRemembered } from './local-discovery'
import { AppConfig } from '@magnitudedev/storage'
import type { ModelsDevResponse, OpenRouterResponse } from './types'

function resolveSource<T>(
  cache: CatalogCacheService,
  sourceId: string,
  ttlMs: number,
  fetchEffect: Effect.Effect<T, Error>,
): Effect.Effect<T | null> {
  return Effect.gen(function* () {
    const cached = yield* cache.load<T>(sourceId).pipe(
      Effect.catchAll(() => Effect.succeed(null)),
    )
    if (cached && cache.isValid(cached)) return cached.data

    const fresh = yield* fetchEffect.pipe(Effect.option)
    if (Option.isSome(fresh)) {
      yield* cache.save(sourceId, fresh.value, ttlMs).pipe(
        Effect.catchAll(() => Effect.void),
      )
      return fresh.value
    }

    if (cached && cache.isStale(cached)) return cached.data
    return null
  })
}

function mergeOpenRouterModel(
  model: ModelDefinition,
  modelsDevModels: readonly ModelDefinition[],
): ModelDefinition {
  const modelsDevModel = modelsDevModels.find((entry) => entry.id === model.id)
  if (!modelsDevModel) return model
  return mergeProviderModels([], [modelsDevModel], [model])[0] ?? model
}

export const ModelCatalogLive = Layer.scoped(
  ModelCatalog,
  Effect.gen(function* () {
    const cache = yield* CatalogCache
    const config = yield* AppConfig

    const refresh: Effect.Effect<void> = Effect.gen(function* () {
      const modelsDevData = yield* resolveSource<ModelsDevResponse>(
        cache,
        'models-dev',
        MODELS_DEV_TTL_MS,
        fetchModelsDevData,
      )
      const openRouterData = yield* resolveSource<OpenRouterResponse>(
        cache,
        'openrouter',
        OPENROUTER_TTL_MS,
        fetchOpenRouterModels,
      )

      for (const provider of PROVIDERS) {
        if (provider.providerFamily === 'local' && provider.inventoryMode === 'dynamic') {
          const options = yield* config.getProviderOptions(provider.id)
          const effectiveBaseUrl = options?.baseUrl?.trim() || provider.defaultBaseUrl
          const discovery = yield* discoverLocalProviderModels(provider, effectiveBaseUrl)
          const remembered = Array.isArray(options?.rememberedModelIds)
            ? options?.rememberedModelIds.filter((id): id is string => typeof id === 'string')
            : []
          const merged = mergeDiscoveredAndRemembered(discovery.models, remembered)
          setProviderModels(provider.id, merged)

          const now = new Date().toISOString()
          yield* config.setProviderOptions(provider.id, (current) => ({
            ...(current ?? {}),
            discoveredModels: discovery.models.map((model) => ({
              id: model.id,
              name: model.name,
              maxContextTokens: model.maxContextTokens ?? null,
              discoveredAt: now,
              source: discovery.source ?? provider.localDiscoveryStrategy ?? 'unknown',
            })),
            inventoryUpdatedAt: now,
            lastDiscoveryStatus: discovery.status,
            lastDiscoverySource: discovery.source ?? undefined,
            lastDiscoveryDiagnostics: discovery.diagnostics.length > 0 ? discovery.diagnostics : undefined,
            ...(discovery.error ? { lastDiscoveryError: discovery.error } : { lastDiscoveryError: undefined }),
          }))

          continue
        }

        const staticModels = [...getStaticProviderModels(provider.id)]

        if (provider.id === 'openrouter') {
          const openRouterModels = openRouterData
            ? normalizeOpenRouterModels(openRouterData)
            : []
          const modelsDevModels = modelsDevData
            ? normalizeModelsDevProvider('openrouter', modelsDevData)
            : []

          if (openRouterModels.length > 0) {
            setProviderModels(
              provider.id,
              openRouterModels.map((model) => mergeOpenRouterModel(model, modelsDevModels)),
            )
          } else if (modelsDevModels.length > 0) {
            setProviderModels(provider.id, mergeProviderModels(staticModels, modelsDevModels))
          } else if (staticModels.length > 0) {
            setProviderModels(provider.id, [...staticModels])
          }

          continue
        }

        const modelsDevModels = modelsDevData
          ? normalizeModelsDevProvider(provider.id, modelsDevData)
          : []

        if (modelsDevModels.length > 0) {
          setProviderModels(provider.id, mergeProviderModels(staticModels, modelsDevModels))
        } else if (staticModels.length > 0) {
          setProviderModels(provider.id, [...staticModels])
        }
      }
    })

    yield* makeRefreshSchedule(refresh)
    yield* refresh

    const getModels = (
      providerId: string,
    ): Effect.Effect<readonly ModelDefinition[]> =>
      Effect.sync(() => getProvider(providerId)?.models ?? [])

    return {
      refresh: () => refresh,
      getModels,
    }
  }),
)

export * from './cache'
export * from './contracts'
export * from './merge'
export * from './models-dev-source'
export * from './openrouter-source'
export * from './refresh'
export * from './types'