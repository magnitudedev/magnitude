import { AppConfig, CatalogCache } from '@magnitudedev/storage'
import { Effect, Layer } from 'effect'

import { ProviderAuth } from '../runtime/contracts'
import { CatalogSourceRegistry, ModelCatalog } from './contracts'
import { mergeProviderModels } from './merge'
import { makeRefreshSchedule } from './refresh'
import { getProvider, getStaticProviderModels, PROVIDERS, setProviderModels } from '../registry'
import type { ModelDefinition } from '../types'

export const ModelCatalogLive = Layer.scoped(
  ModelCatalog,
  Effect.gen(function* () {
    const sourceRegistry = yield* CatalogSourceRegistry
    const catalogCache = yield* CatalogCache
    const appConfig = yield* AppConfig
    const providerAuth = yield* ProviderAuth

    const refresh = Effect.gen(function* () {
      const sources = sourceRegistry.list()

      for (const provider of PROVIDERS) {
        const matchingSources = sources
          .filter((source) => source.supports(provider))
          .sort((a, b) => a.priority - b.priority)

        const successfulLayers: ModelDefinition[][] = []

        for (const source of matchingSources) {
          const result = yield* Effect.either(source.refresh(provider))
          if (result._tag === 'Right') {
            successfulLayers.push([...result.right])
          } else {
            yield* Effect.logDebug(`Catalog source ${source.id} failed for ${provider.id}`, {
              error: result.left,
            })
          }
        }

        const merged = successfulLayers.length > 0
          ? mergeProviderModels([], ...successfulLayers)
          : [...getStaticProviderModels(provider.id)]

        setProviderModels(provider.id, merged)
      }
    }).pipe(
      Effect.provideService(CatalogCache, catalogCache),
      Effect.provideService(AppConfig, appConfig),
      Effect.provideService(ProviderAuth, providerAuth),
    )

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
export * from './local-catalog-source'
export * from './merge'
export * from './models-dev-catalog-source'
export * from './openrouter-catalog-source'
export * from './refresh'
export * from './source-registry'
export * from './static-catalog-source'
export * from './types'