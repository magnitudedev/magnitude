import { Effect } from 'effect'
import { afterEach, describe, expect, it, mock } from 'bun:test'
import { discoverLocalProviderModels, discoverOllamaHybrid, discoverOpenAIModels, mergeDiscoveredAndRemembered } from '../catalog/local-discovery'
import { getProvider } from '../registry'

const originalFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = originalFetch
})

describe('local discovery', () => {
  it('discovers openai models from /v1/models', async () => {
    globalThis.fetch = mock(async () =>
      new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:14b' }] }), { status: 200 }),
    ) as any

    const models = await Effect.runPromise(discoverOpenAIModels('http://localhost:1234/v1'))
    expect(models.map((m) => m.id)).toEqual(['qwen2.5-coder:14b'])
  })

  it('returns structured discovery error instead of silently collapsing to zero models', async () => {
    globalThis.fetch = mock(async () => new Response('nope', { status: 500 })) as any
    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')

    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))
    expect(result.models).toEqual([])
    expect(typeof result.error).toBe('string')
    expect(result.error).toContain('LM Studio discovery failed')
    expect(result.error).toContain('/v1/models')
    expect(result.error).toContain('/api/v1/models')
  })

  it('lmstudio falls back to native /api/v1/models when openai endpoint fails', async () => {
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.endsWith('/api/v1/models')) return new Response(JSON.stringify({ data: [{ id: 'qwen-native', name: 'Qwen Native' }] }), { status: 200 })
      if (raw.endsWith('/v1/models')) return new Response('bad', { status: 500 })
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.source).toBe('lmstudio-native-v1-models')
    expect(result.models.map((m) => m.id)).toEqual(['qwen-native'])
  })

  it('lmstudio falls back to legacy /api/v0/models when openai and native v1 fail', async () => {
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.endsWith('/v1/models')) return new Response('bad', { status: 500 })
      if (raw.endsWith('/api/v1/models')) return new Response('bad', { status: 500 })
      if (raw.endsWith('/api/v0/models')) return new Response(JSON.stringify({ models: [{ model: 'legacy-model', name: 'Legacy' }] }), { status: 200 })
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.source).toBe('lmstudio-native-v0-models')
    expect(result.models.map((m) => m.id)).toEqual(['legacy-model'])
  })

  it('returns success_empty (not failure) when lmstudio endpoints succeed with zero models', async () => {
    globalThis.fetch = mock(async (_url: string | URL | Request) =>
      new Response(JSON.stringify({ data: [] }), { status: 200 }),
    ) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.models).toEqual([])
    expect(result.error).toBeNull()
    expect(result.status).toBe('success_empty')
    expect(result.source).toBe('openai-v1-models')
  })

  it('ollama hybrid uses /api/tags as primary endpoint', async () => {
    const seen: string[] = []
    globalThis.fetch = mock(async (url: string) => {
      seen.push(url)
      if (url.endsWith('/api/tags')) {
        return new Response(JSON.stringify({ models: [{ name: 'llama3.2:3b' }] }), { status: 200 })
      }
      if (url.endsWith('/v1/models')) return new Response('unexpected', { status: 500 })
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models.map((m) => m.id)).toEqual(['llama3.2:3b'])
    expect(seen.some((u) => u.endsWith('/api/tags'))).toBe(true)
    expect(seen.some((u) => u.endsWith('/v1/models'))).toBe(false)
  })

  it('ollama hybrid falls back to /v1/models when /api/tags fails', async () => {
    globalThis.fetch = mock(async (url: string) => {
      if (url.endsWith('/api/tags')) return new Response('bad', { status: 500 })
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'llama3.2:3b' }] }), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models.map((m) => m.id)).toEqual(['llama3.2:3b'])
  })

  it('merges discovered and remembered model ids without duplicates', () => {
    const merged = mergeDiscoveredAndRemembered(
      [
        {
          id: 'model-a',
          name: 'model-a',
          contextWindow: 200_000,
          supportsToolCalls: true,
          supportsReasoning: false,
          cost: { input: 0, output: 0 },
          family: 'local',
          releaseDate: new Date().toISOString(),
          discovery: { primarySource: 'static' },
        },
      ],
      ['model-a', 'model-b'],
    )

    expect(merged.map((m) => m.id)).toEqual(['model-a', 'model-b'])
  })
})
