import { describe, expect, it, mock, afterEach } from 'bun:test'
import { Effect } from 'effect'
import { createStorageClient } from '@magnitudedev/storage'
import { ModelCatalogLive, ModelCatalog } from '../catalog'
import { getProvider } from '../registry'

const originalFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = originalFetch
})

describe('local catalog discovery metadata persistence', () => {
  it('persists discoveredModels and clears lastDiscoveryError on success', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
      lastDiscoveryError: 'old error',
    })

    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.includes('localhost:1234') && raw.endsWith('/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:14b', name: 'Qwen' }] }), { status: 200 })
      }
      return new Response('unavailable', { status: 503 })
    }) as any

    await Effect.runPromise(
      Effect.gen(function* () {
        const catalog = yield* ModelCatalog
        yield* catalog.refresh()
      }).pipe(
        Effect.provide(ModelCatalogLive),
        Effect.provide(storage.layer),
      ),
    )

    const options = await storage.config.getProviderOptions('lmstudio')
    expect(options?.inventoryUpdatedAt).toBeDefined()
    expect(options?.lastDiscoveryError).toBeUndefined()
    expect(options?.lastDiscoveryStatus).toBe('success_non_empty')
    expect(options?.lastDiscoverySource).toBe('openai-v1-models')
    expect(options?.discoveredModels?.map((m) => m.id)).toEqual(['qwen2.5-coder:14b'])
    expect(options?.discoveredModels?.[0]?.source).toBe('openai-v1-models')

    const provider = getProvider('lmstudio')
    expect(provider?.models.map((m) => m.id)).toEqual(['qwen2.5-coder:14b'])
  })

  it('persists lastDiscoveryError when discovery fails', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
    })

    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.includes('localhost:1234') && raw.endsWith('/models')) {
        return new Response('nope', { status: 500 })
      }
      return new Response('unavailable', { status: 503 })
    }) as any

    await Effect.runPromise(
      Effect.gen(function* () {
        const catalog = yield* ModelCatalog
        yield* catalog.refresh()
      }).pipe(
        Effect.provide(ModelCatalogLive),
        Effect.provide(storage.layer),
      ),
    )

    const options = await storage.config.getProviderOptions('lmstudio')
    expect(options?.inventoryUpdatedAt).toBeDefined()
    expect(options?.discoveredModels ?? []).toEqual([])
    expect(typeof options?.lastDiscoveryError).toBe('string')
    expect(options?.lastDiscoveryError).toContain('HTTP 500')
    expect(options?.lastDiscoveryStatus).toBe('failure')
    expect(Array.isArray(options?.lastDiscoveryDiagnostics)).toBe(true)
    expect((options?.lastDiscoveryDiagnostics ?? []).length).toBeGreaterThan(0)
  })
})
