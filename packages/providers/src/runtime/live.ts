import { Effect, Layer } from 'effect'
import { ProviderCatalog, ProviderState, ProviderConfig, ProviderAuth } from './contracts'
import { PROVIDERS, getProvider } from '../registry'
import { detectDefaultProvider, detectProviderAuthMethods, detectProviders } from '../detect'
import {
  loadAuth,
  getAuth,
  setAuth,
  removeAuth,
  loadConfig,
  saveConfig,
  setPrimarySelection,
  setBrowserSelection,
} from '../config'
import { getLocalProviderConfig, setLocalProviderConfig } from '../local-config'
import {
  peekSlot,
  getSlots,
  setModel,
  clearModel,
  getModelContextWindow,
  getSlotUsage,
  accumulateUsage,
  resetSlotUsage,
} from '../state/provider-state'
import { refreshAnthropicToken } from '../auth/anthropic-oauth'
import { refreshOpenAIToken } from '../auth/openai-oauth'
import { exchangeCopilotToken } from '../auth/copilot-oauth'
import { initializeModels } from '../models-dev'

const COMPACT_TRIGGER_RATIO = 0.9

export function makeProviderRuntimeLive() {
  return Layer.mergeAll(
    Layer.succeed(ProviderCatalog, {
      listProviders: () => Effect.succeed(PROVIDERS),
      getProvider: (providerId: string) => Effect.succeed(getProvider(providerId) ?? null),
      getProviderName: (providerId: string) => Effect.succeed(getProvider(providerId)?.name ?? providerId),
      listModels: (providerId: string) => Effect.succeed(getProvider(providerId)?.models ?? []),
      getModel: (providerId: string, modelId: string) =>
        Effect.succeed(getProvider(providerId)?.models.find((m) => m.id === modelId) ?? null),
      refresh: () => Effect.promise(() => initializeModels()),
    }),
    Layer.succeed(ProviderState, {
      peek: (slot = 'primary') => Effect.succeed(peekSlot(slot)),
      getSlot: (slot) => Effect.succeed(getSlots()[slot]),
      setSelection: (slot, providerId, modelId, auth, options) =>
        Effect.succeed(setModel(slot, providerId, modelId, auth, options?.persist ?? true)),
      clear: (slot) => Effect.sync(() => clearModel(slot)),
      contextWindow: (slot = 'primary') => Effect.succeed(getModelContextWindow(slot)),
      contextLimits: (slot = 'primary') =>
        Effect.sync(() => {
          const hardCap = getModelContextWindow(slot)
          return { hardCap, softCap: Math.floor(hardCap * COMPACT_TRIGGER_RATIO) }
        }),
      accumulateUsage: (slot, usage) => Effect.sync(() => accumulateUsage(slot, usage)),
      getUsage: (slot) => Effect.succeed(getSlotUsage(slot)),
      resetUsage: (slot) => Effect.sync(() => resetSlotUsage(slot)),
    }),
    Layer.succeed(ProviderConfig, {
      loadConfig: () => Effect.succeed(loadConfig()),
      saveConfig: (config) => Effect.sync(() => saveConfig(config)),
      setPrimarySelection: (providerId, modelId) => Effect.sync(() => setPrimarySelection(providerId, modelId)),
      setBrowserSelection: (providerId, modelId) => Effect.sync(() => setBrowserSelection(providerId, modelId)),
      getLocalProviderConfig: () => Effect.succeed(getLocalProviderConfig()),
      setLocalProviderConfig: (baseUrl, modelId) => Effect.sync(() => setLocalProviderConfig(baseUrl, modelId)),
    }),
    Layer.succeed(ProviderAuth, {
      loadAuth: () => Effect.succeed(loadAuth()),
      getAuth: (providerId) => Effect.succeed(getAuth(providerId)),
      setAuth: (providerId, auth) => Effect.sync(() => setAuth(providerId, auth)),
      removeAuth: (providerId) => Effect.sync(() => removeAuth(providerId)),
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
        Effect.succeed(
          detectProviders().map((entry) => ({
            provider: entry.provider,
            authMethods: detectProviderAuthMethods(entry.provider.id)?.methods ?? [],
          })),
        ),
      detectDefaultProvider: () => Effect.succeed(detectDefaultProvider()?.provider.id ?? null),
      detectProviderAuthMethods: (providerId) => Effect.succeed(detectProviderAuthMethods(providerId)),
      connectedProviderIds: () => Effect.succeed(new Set(detectProviders().map((d) => d.provider.id))),
    }),
  )
}