import { Effect, Layer } from 'effect'
import {
  AppConfig,
  AuthStorage,
  AuthStorageLive,
  CatalogCacheLive,
  ConfigStorageLive,
  GlobalStorageLive,
  computeContextLimits,
} from '@magnitudedev/storage'
import { ProviderCatalog, ProviderState, ProviderAuth } from './contracts'
import { PROVIDERS, getProvider } from '../registry'
import { detectDefaultProvider, detectProviderAuthMethods, detectProviders } from '../detect'
import { ModelCatalog, ModelCatalogLive } from '../catalog'

import { makeProviderStateStore } from '../state/provider-state'
import { refreshAnthropicToken } from '../auth/anthropic-oauth'
import { refreshOpenAIToken } from '../auth/openai-oauth'
import { exchangeCopilotToken } from '../auth/copilot-oauth'

export function makeProviderRuntimeLive<TSlot extends string>() {
  const stateStore = makeProviderStateStore<TSlot>()

  const storageLayer = Layer.mergeAll(
    Layer.provide(ConfigStorageLive, GlobalStorageLive),
    Layer.provide(AuthStorageLive, GlobalStorageLive),
    Layer.provide(CatalogCacheLive, GlobalStorageLive),
  )

  const catalogLayer = Layer.provide(ModelCatalogLive, storageLayer)

  const providerCatalogLayer = Layer.effect(
    ProviderCatalog,
    Effect.gen(function* () {
      const catalog = yield* ModelCatalog
      return {
        listProviders: () => Effect.succeed(PROVIDERS),
        getProvider: (providerId: string) => Effect.succeed(getProvider(providerId) ?? null),
        getProviderName: (providerId: string) => Effect.succeed(getProvider(providerId)?.name ?? providerId),
        listModels: (providerId: string) => catalog.getModels(providerId),
        getModel: (providerId: string, modelId: string) =>
          Effect.map(catalog.getModels(providerId), (models) => models.find((model) => model.id === modelId) ?? null),
        refresh: () => catalog.refresh(),
      }
    }),
  )

  const providerLayer = Layer.provide(
    Layer.mergeAll(
      providerCatalogLayer,
      Layer.effect(
        ProviderState,
        Effect.gen(function* () {
          const config = yield* AppConfig
          return {
            peek: (slot) => Effect.succeed(stateStore.peekSlot(slot as TSlot)),
            getSlot: (slot) => Effect.succeed(stateStore.getSlots().get(slot as TSlot) ?? {
              providerId: null,
              modelId: null,
              auth: null,
              registry: undefined,
            }),
            setSelection: (slot, providerId, modelId, auth, options) =>
              Effect.flatMap(config.getProviderOptions(providerId), (providerOptions) =>
                Effect.flatMap(
                  Effect.sync(() => stateStore.setModel(slot as TSlot, providerId, modelId, auth, providerOptions)),
                  (ok) => {
                    if (!ok || options?.persist === false) return Effect.succeed(ok)
                    return Effect.as(config.setModelSelection(slot, { providerId, modelId }), ok)
                  },
                ),
              ),
            clear: (slot) => Effect.sync(() => stateStore.clearModel(slot as TSlot)),
            contextWindow: (slot) => Effect.succeed(stateStore.getModelContextWindow(slot as TSlot)),
            contextLimits: (slot) =>
              Effect.map(config.getContextLimitPolicy(), (policy) =>
                computeContextLimits(stateStore.getModelContextWindow(slot as TSlot), policy),
              ),
            accumulateUsage: (slot, usage) => Effect.sync(() => stateStore.accumulateUsage(slot as TSlot, usage)),
            getUsage: (slot) => Effect.succeed(stateStore.getSlotUsage(slot as TSlot)),
            resetUsage: (slot) => Effect.sync(() => stateStore.resetSlotUsage(slot as TSlot)),
          }
        }),
      ),
      Layer.effect(
        ProviderAuth,
        Effect.gen(function* () {
          const authStorage = yield* AuthStorage
          const config = yield* AppConfig
          return {
            loadAuth: () => authStorage.loadAll(),
            getAuth: (providerId) => authStorage.get(providerId),
            setAuth: (providerId, auth) => authStorage.set(providerId, auth),
            removeAuth: (providerId) => authStorage.remove(providerId),
            refresh: (providerId, refreshToken) =>
              Effect.tryPromise({
                try: async () => {
                  if (providerId === 'anthropic') return refreshAnthropicToken(refreshToken)
                  if (providerId === 'openai') return refreshOpenAIToken(refreshToken)
                  if (providerId === 'github-copilot') return exchangeCopilotToken(refreshToken)
                  return null
                },
                catch: (error) => (error instanceof Error ? error : new Error(String(error))),
              }),
            detectProviders: () =>
              Effect.flatMap(authStorage.loadAll(), (storedAuth) =>
                Effect.flatMap(config.getLocalProviderConfig(), (localProviderConfig) =>
                  Effect.succeed(
                    detectProviders(storedAuth, localProviderConfig).map((entry) => ({
                      provider: entry.provider,
                      authMethods: detectProviderAuthMethods(entry.provider.id, storedAuth, localProviderConfig)?.methods ?? [],
                    })),
                  ),
                ),
              ),
            detectDefaultProvider: () =>
              Effect.flatMap(authStorage.loadAll(), (storedAuth) =>
                Effect.map(
                  config.getLocalProviderConfig(),
                  (localProviderConfig) => detectDefaultProvider(storedAuth, localProviderConfig)?.provider.id ?? null,
                ),
              ),
            detectProviderAuthMethods: (providerId) =>
              Effect.flatMap(authStorage.loadAll(), (storedAuth) =>
                Effect.map(
                  config.getLocalProviderConfig(),
                  (localProviderConfig) => detectProviderAuthMethods(providerId, storedAuth, localProviderConfig),
                ),
              ),
            connectedProviderIds: () =>
              Effect.flatMap(authStorage.loadAll(), (storedAuth) =>
                Effect.map(
                  config.getLocalProviderConfig(),
                  (localProviderConfig) => new Set(detectProviders(storedAuth, localProviderConfig).map((d) => d.provider.id)),
                ),
              ),
          }
        }),
      ),
    ),
    Layer.mergeAll(storageLayer, catalogLayer),
  )

  return Layer.mergeAll(storageLayer, providerLayer)
}