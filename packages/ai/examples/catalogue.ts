import { Effect, Layer } from "effect"
import {
  CatalogueCache,
  CatalogueConfig,
  ModelCatalogue,
  ModelCatalogueLive,
  type CachedData,
  type ProviderOptions,
} from "../src/index.js"

const cacheState = new Map<string, CachedData<unknown>>()
const providerOptionsState = new Map<string, ProviderOptions | undefined>()

const CacheLive = Layer.succeed(CatalogueCache, {
  load: <T>(sourceId: string) => Effect.succeed((cacheState.get(sourceId) as CachedData<T> | undefined) ?? null),
  save: (sourceId: string, data: unknown, ttlMs: number) =>
    Effect.sync(() => {
      cacheState.set(sourceId, {
        _cachedAt: Date.now(),
        ttlMs,
        data,
      })
    }),
  isValid: (cached: CachedData<unknown>) => Date.now() - cached._cachedAt <= cached.ttlMs,
  isStale: (cached: CachedData<unknown>) => Date.now() - cached._cachedAt > cached.ttlMs,
})

const ConfigLive = Layer.succeed(CatalogueConfig, {
  getProviderOptions: (providerId: string) =>
    Effect.succeed(providerOptionsState.get(providerId)),
  setProviderOptions: (
    providerId: string,
    optionsOrUpdater:
      | ProviderOptions
      | undefined
      | ((current: ProviderOptions | undefined) => ProviderOptions | undefined),
  ) =>
    Effect.sync(() => {
      const current = providerOptionsState.get(providerId)
      const next =
        typeof optionsOrUpdater === "function"
          ? optionsOrUpdater(current)
          : optionsOrUpdater
      providerOptionsState.set(providerId, next)
    }),
})

const program = Effect.gen(function* () {
  const catalogue = yield* ModelCatalogue

  yield* catalogue.refresh()

  const openAiModels = yield* catalogue.getModels("openai")
  console.log(`OpenAI models: ${openAiModels.length}`)

  const allModels = yield* catalogue.getAllModels()
  console.log(`Providers in catalogue: ${allModels.size}`)
})

const CatalogueLive = Layer.mergeAll(CacheLive, ConfigLive, ModelCatalogueLive)

program.pipe(Effect.provide(CatalogueLive), Effect.runPromise)
