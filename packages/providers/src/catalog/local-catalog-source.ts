import { AppConfig } from '@magnitudedev/storage'
import { Effect } from 'effect'

import type { CatalogSource } from './contracts'
import { discoverLocalProviderModels, mergeDiscoveredAndRemembered } from './local-discovery'

export const localCatalogSource: CatalogSource = {
  id: 'local-discovery',
  priority: 200,
  supports: (provider) =>
    provider.providerFamily === 'local' && provider.inventoryMode === 'dynamic',
  refresh: (provider) =>
    Effect.gen(function* () {
      const config = yield* AppConfig
      const options = yield* config.getProviderOptions(provider.id)
      const effectiveBaseUrl = options?.baseUrl?.trim() || provider.defaultBaseUrl
      const discovery = yield* discoverLocalProviderModels(provider, effectiveBaseUrl)

      const remembered = Array.isArray(options?.rememberedModelIds)
        ? options.rememberedModelIds.filter((id): id is string => typeof id === 'string')
        : []
      const merged = mergeDiscoveredAndRemembered(discovery.models, remembered)

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

      return merged
    }),
}
