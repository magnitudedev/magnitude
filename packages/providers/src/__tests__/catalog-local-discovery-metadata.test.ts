import { afterEach, describe, expect, it, vi } from 'vitest'
import { Effect } from 'effect'
import { createStorageClient } from '@magnitudedev/storage'
import { ModelCatalogLive, ModelCatalog, CatalogSourceRegistry } from '../catalog'
import { localCatalogSource } from '../catalog/local-catalog-source'
import { staticCatalogSource } from '../catalog/static-catalog-source'
import { getProvider } from '../registry'
import { ProviderAuth, type ProviderAuthShape } from '../runtime/contracts'

const originalFetch = globalThis.fetch

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

function stubFetch(
  impl: (...args: Parameters<typeof fetch>) => ReturnType<typeof fetch>,
): void {
  const mock = vi.fn(impl)
  const fetchStub: typeof fetch = Object.assign(
    (...args: Parameters<typeof fetch>) => mock(...args),
    originalFetch,
  )
  globalThis.fetch = fetchStub
}

async function runRefresh(storage: Awaited<ReturnType<typeof createStorageClient>>) {
  await Effect.runPromise(
    Effect.gen(function* () {
      const catalog = yield* ModelCatalog
      yield* catalog.refresh()
    }).pipe(
      Effect.provide(ModelCatalogLive),
      Effect.provideService(CatalogSourceRegistry, {
        list: () => [
          staticCatalogSource,
          localCatalogSource,
        ],
      }),
      Effect.provideService(ProviderAuth, providerAuthStub),
      Effect.provide(storage.layer),
    ),
  )
}

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

    stubFetch(async (url) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:14b', max_context_length: 32768 }] }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v0/models')) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:14b', name: 'Qwen' }] }), { status: 200 })
      }
      return new Response('unavailable', { status: 503 })
    })

    await runRefresh(storage)

    const options = await storage.config.getProviderOptions('lmstudio')
    expect(options?.inventoryUpdatedAt).toBeDefined()
    expect(options?.lastDiscoveryError).toBeUndefined()
    expect(options?.lastDiscoveryStatus).toBe('success_non_empty')
    expect(options?.lastDiscoverySource).toBe('openai-v1-models')
    expect(options?.discoveredModels?.map((m) => m.id)).toEqual(['qwen2.5-coder:14b'])
    expect(options?.discoveredModels?.[0]?.source).toBe('openai-v1-models')
    expect(options?.discoveredModels?.[0]?.maxContextTokens).toBe(32768)

    const provider = getProvider('lmstudio')
    expect(provider?.models.map((m) => m.id)).toEqual(['qwen2.5-coder:14b'])
  })

  it('persists lastDiscoveryError when discovery fails', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
    })

    stubFetch(async (url) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.includes('localhost:1234') && raw.endsWith('/models')) {
        return new Response('nope', { status: 500 })
      }
      return new Response('unavailable', { status: 503 })
    })

    await runRefresh(storage)

    const options = await storage.config.getProviderOptions('lmstudio')
    expect(options?.inventoryUpdatedAt).toBeDefined()
    expect(options?.discoveredModels ?? []).toEqual([])
    expect(typeof options?.lastDiscoveryError).toBe('string')
    expect(options?.lastDiscoveryError).toContain('HTTP 500')
    expect(options?.lastDiscoveryStatus).toBe('failure')
    expect(Array.isArray(options?.lastDiscoveryDiagnostics)).toBe(true)
    expect((options?.lastDiscoveryDiagnostics ?? []).length).toBeGreaterThan(0)
  })

  it('overwrites persisted maxContextTokens on refresh for same discovered model id', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
    })

    let nativeContext = 32768

    stubFetch(async (url) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:14b', max_context_length: nativeContext }] }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v0/models')) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:14b', name: 'Qwen' }] }), { status: 200 })
      }
      return new Response('unavailable', { status: 503 })
    })

    await runRefresh(storage)

    const first = await storage.config.getProviderOptions('lmstudio')
    expect(first?.discoveredModels?.[0]?.id).toBe('qwen2.5-coder:14b')
    expect(first?.discoveredModels?.[0]?.maxContextTokens).toBe(32768)

    nativeContext = 8192

    await runRefresh(storage)

    const second = await storage.config.getProviderOptions('lmstudio')
    expect(second?.discoveredModels?.[0]?.id).toBe('qwen2.5-coder:14b')
    expect(second?.discoveredModels?.[0]?.maxContextTokens).toBe(8192)
  })

  it('persists loaded runtime context with bounded LM Studio id reconciliation (trim + case-fold)', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
    })

    let loadedContext = 73728

    stubFetch(async (url) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({
          models: [{
            id: '  GEMMA-3-12B-IT  ',
            max_context_length: 131072,
            loaded_instances: [{ config: { context_length: loadedContext } }],
          }],
        }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({
          data: [{ id: 'gemma-3-12b-it', name: 'Gemma 3 12B IT' }],
        }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v0/models')) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 })
      }
      return new Response('unavailable', { status: 503 })
    })

    await runRefresh(storage)

    const first = await storage.config.getProviderOptions('lmstudio')
    expect(first?.discoveredModels?.[0]?.id).toBe('gemma-3-12b-it')
    expect(first?.discoveredModels?.[0]?.maxContextTokens).toBe(73728)

    loadedContext = 65536

    await runRefresh(storage)

    const second = await storage.config.getProviderOptions('lmstudio')
    expect(second?.discoveredModels?.[0]?.id).toBe('gemma-3-12b-it')
    expect(second?.discoveredModels?.[0]?.maxContextTokens).toBe(65536)
  })

  it('persists v0 loaded_context_length when v1 only provides static max', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
    })

    let loadedContext = 73195

    stubFetch(async (url) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.includes('localhost:1234') && raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'gemma-3-12b-it' }] }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({
          models: [{ id: 'gemma-3-12b-it', max_context_length: 131072 }],
        }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v0/models')) {
        return new Response(JSON.stringify({
          models: [{ id: 'gemma-3-12b-it', loaded_context_length: loadedContext, max_context_length: 131072 }],
        }), { status: 200 })
      }
      return new Response('unavailable', { status: 503 })
    })

    await runRefresh(storage)

    const first = await storage.config.getProviderOptions('lmstudio')
    expect(first?.discoveredModels?.[0]?.maxContextTokens).toBe(73195)

    loadedContext = 65536

    await runRefresh(storage)

    const second = await storage.config.getProviderOptions('lmstudio')
    expect(second?.discoveredModels?.[0]?.maxContextTokens).toBe(65536)
  })

  it('persists unknown discovered context as null', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
    })

    stubFetch(async (url) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:14b' }] }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/api/v0/models')) {
        return new Response(JSON.stringify({ models: [] }), { status: 200 })
      }
      if (raw.includes('localhost:1234') && raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:14b', name: 'Qwen' }] }), { status: 200 })
      }
      return new Response('unavailable', { status: 503 })
    })

    await runRefresh(storage)

    const options = await storage.config.getProviderOptions('lmstudio')
    expect(options?.discoveredModels?.[0]?.id).toBe('qwen2.5-coder:14b')
    expect(options?.discoveredModels?.[0]?.maxContextTokens).toBeNull()
  })
})
