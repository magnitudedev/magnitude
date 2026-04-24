import { createStorageClient } from '@magnitudedev/storage'
import { Effect, Layer } from 'effect'
import { describe, expect, it } from 'vitest'

import {
  CatalogSourceRegistry,
  ModelCatalog,
  ModelCatalogLive,
  type CatalogSource,
  CatalogTransportError,
} from '../catalog'
import { ProviderAuth } from '../runtime/contracts'
import type { ProviderAuthShape } from '../runtime/contracts'
import { getProvider } from '../registry'
import type { ModelDefinition } from '../types'

const providerAuthStub: ProviderAuthShape = {
  loadAuth: () => Effect.succeed({}),
  getAuth: () => Effect.succeed(undefined),
  setAuth: () => Effect.void,
  removeAuth: () => Effect.void,
  refresh: () => Effect.succeed(null),
  detectProviders: () => Effect.succeed([]),
  detectDefaultProvider: () => Effect.succeed(null),
  detectProviderAuthMethods: () => Effect.succeed(null),
  connectedProviderIds: () => Effect.succeed(new Set()),
}

function model(id: string, overrides: Partial<ModelDefinition> = {}): ModelDefinition {
  return {
    id,
    name: id,
    contextWindow: 1000,
    supportsToolCalls: false,
    supportsReasoning: false,
    cost: { input: 0, output: 0 },
    family: 'test',
    releaseDate: '2025-01-01T00:00:00.000Z',
    discovery: { primarySource: 'static' },
    ...overrides,
  }
}

describe('ModelCatalog source refresh', () => {
  it('keeps static Fire Pass when higher-priority fireworks-ai source fails', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    const fireworks = getProvider('fireworks-ai')!

    const sources: CatalogSource[] = [
      {
        id: 'static',
        priority: 0,
        supports: (provider) => provider.id === 'fireworks-ai',
        refresh: () => Effect.succeed([
          model('accounts/fireworks/models/kimi-k2p6'),
        ]),
      },
      {
        id: 'fireworks-api',
        priority: 300,
        supports: (provider) => provider.id === 'fireworks-ai',
        refresh: () => Effect.fail(new CatalogTransportError({
          sourceId: 'fireworks-api',
          providerId: 'fireworks-ai',
          message: 'missing auth',
        })),
      },
    ]

    const layer = Layer.provide(
      ModelCatalogLive,
      Layer.succeed(CatalogSourceRegistry, { list: () => sources }),
    )

    const catalog = await Effect.runPromise(
      Effect.provide(ModelCatalog, layer).pipe(
        Effect.provideService(ProviderAuth, providerAuthStub),
        Effect.provide(storage.layer),
      ),
    )

    await Effect.runPromise(catalog.refresh())
    const models = await Effect.runPromise(catalog.getModels(fireworks.id))

    expect(models.some((entry) => entry.id === 'accounts/fireworks/models/kimi-k2p6')).toBe(true)
  })

  it('overlays successful source layers in priority order', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    const openrouter = getProvider('openrouter')!

    const sources: CatalogSource[] = [
      {
        id: 'static',
        priority: 0,
        supports: (provider) => provider.id === 'openrouter',
        refresh: () => Effect.succeed([model('anthropic/claude-opus-4.6', { name: 'Static' })]),
      },
      {
        id: 'models.dev',
        priority: 100,
        supports: (provider) => provider.id === 'openrouter',
        refresh: () => Effect.succeed([model('anthropic/claude-opus-4.6', { description: 'Enriched', discovery: { primarySource: 'models.dev' } })]),
      },
      {
        id: 'openrouter-api',
        priority: 300,
        supports: (provider) => provider.id === 'openrouter',
        refresh: () => Effect.succeed([model('anthropic/claude-opus-4.6', { name: 'Live', supportsToolCalls: true, discovery: { primarySource: 'openrouter-api' } })]),
      },
    ]

    const layer = Layer.provide(
      ModelCatalogLive,
      Layer.succeed(CatalogSourceRegistry, { list: () => sources }),
    )

    const catalog = await Effect.runPromise(
      Effect.provide(ModelCatalog, layer).pipe(
        Effect.provideService(ProviderAuth, providerAuthStub),
        Effect.provide(storage.layer),
      ),
    )

    await Effect.runPromise(catalog.refresh())
    const models = await Effect.runPromise(catalog.getModels(openrouter.id))
    const selected = models.find((entry) => entry.id === 'anthropic/claude-opus-4.6')

    expect(selected?.name).toBe('Live')
    expect(selected?.description).toBe('Enriched')
    expect(selected?.supportsToolCalls).toBe(true)
  })

  it('does not change unrelated provider inventory when fireworks refresh succeeds', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    const fireworks = getProvider('fireworks-ai')!
    const anthropic = getProvider('anthropic')!
    const anthropicBefore = anthropic.models.map((entry) => entry.id)

    const sources: CatalogSource[] = [
      {
        id: 'static',
        priority: 0,
        supports: (provider) => provider.id === 'anthropic' || provider.id === 'fireworks-ai',
        refresh: (provider) => Effect.succeed(provider.models),
      },
      {
        id: 'fireworks-api',
        priority: 300,
        supports: (provider) => provider.id === 'fireworks-ai',
        refresh: () => Effect.succeed([
          model('accounts/fireworks/models/kimi-k2p5', {
            name: 'Kimi K2.5',
            discovery: { primarySource: 'models.dev' },
          }),
        ]),
      },
    ]

    const layer = Layer.provide(
      ModelCatalogLive,
      Layer.succeed(CatalogSourceRegistry, { list: () => sources }),
    )

    const catalog = await Effect.runPromise(
      Effect.provide(ModelCatalog, layer).pipe(
        Effect.provideService(ProviderAuth, providerAuthStub),
        Effect.provide(storage.layer),
      ),
    )

    await Effect.runPromise(catalog.refresh())

    const fireworksModels = await Effect.runPromise(catalog.getModels(fireworks.id))
    const anthropicModels = await Effect.runPromise(catalog.getModels(anthropic.id))

    expect(fireworksModels.some((entry) => entry.id === 'accounts/fireworks/models/kimi-k2p5')).toBe(true)
    expect(anthropicModels.map((entry) => entry.id)).toEqual(anthropicBefore)
  })
})
