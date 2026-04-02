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
    expect(options?.discoveredModels?.[0]?.maxContextTokens).toBe(32768)

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

  it('overwrites persisted maxContextTokens on refresh for same discovered model id', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
    })

    let nativeContext = 32768

    globalThis.fetch = mock(async (url: string | URL | Request) => {
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

    const first = await storage.config.getProviderOptions('lmstudio')
    expect(first?.discoveredModels?.[0]?.id).toBe('qwen2.5-coder:14b')
    expect(first?.discoveredModels?.[0]?.maxContextTokens).toBe(32768)

    nativeContext = 8192

    await Effect.runPromise(
      Effect.gen(function* () {
        const catalog = yield* ModelCatalog
        yield* catalog.refresh()
      }).pipe(
        Effect.provide(ModelCatalogLive),
        Effect.provide(storage.layer),
      ),
    )

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

    globalThis.fetch = mock(async (url: string | URL | Request) => {
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

    const first = await storage.config.getProviderOptions('lmstudio')
    expect(first?.discoveredModels?.[0]?.id).toBe('gemma-3-12b-it')
    expect(first?.discoveredModels?.[0]?.maxContextTokens).toBe(73728)

    loadedContext = 65536

    await Effect.runPromise(
      Effect.gen(function* () {
        const catalog = yield* ModelCatalog
        yield* catalog.refresh()
      }).pipe(
        Effect.provide(ModelCatalogLive),
        Effect.provide(storage.layer),
      ),
    )

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

    globalThis.fetch = mock(async (url: string | URL | Request) => {
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

    const first = await storage.config.getProviderOptions('lmstudio')
    expect(first?.discoveredModels?.[0]?.maxContextTokens).toBe(73195)

    loadedContext = 65536

    await Effect.runPromise(
      Effect.gen(function* () {
        const catalog = yield* ModelCatalog
        yield* catalog.refresh()
      }).pipe(
        Effect.provide(ModelCatalogLive),
        Effect.provide(storage.layer),
      ),
    )

    const second = await storage.config.getProviderOptions('lmstudio')
    expect(second?.discoveredModels?.[0]?.maxContextTokens).toBe(65536)
  })

  it('persists unknown discovered context as null', async () => {
    const storage = await createStorageClient({ cwd: process.cwd() })
    await storage.config.setProviderOptions('lmstudio', {
      baseUrl: 'http://localhost:1234/v1',
    })

    globalThis.fetch = mock(async (url: string | URL | Request) => {
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
    expect(options?.discoveredModels?.[0]?.id).toBe('qwen2.5-coder:14b')
    expect(options?.discoveredModels?.[0]?.maxContextTokens).toBeNull()
  })
})
