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
      if (raw.endsWith('/api/v1/models')) return new Response(JSON.stringify({ data: [{ id: 'qwen-native', name: 'Qwen Native', max_context_length: 16384 }] }), { status: 200 })
      if (raw.endsWith('/v1/models')) return new Response('bad', { status: 500 })
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.source).toBe('lmstudio-native-v1-models')
    expect(result.models.map((m) => m.id)).toEqual(['qwen-native'])
    expect(result.models[0]?.maxContextTokens).toBe(16384)
  })

  it('lmstudio context precedence prefers loaded_instances context_length over max_context_length', async () => {
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({
          data: [{
            id: 'gemma-local',
            max_context_length: 32768,
            loaded_instances: [{ config: { context_length: 8192 } }],
          }],
        }), { status: 200 })
      }
      if (raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'gemma-local' }] }), { status: 200 })
      }
      if (raw.endsWith('/api/v0/models')) return new Response(JSON.stringify({ data: [] }), { status: 200 })
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.models[0]?.maxContextTokens).toBe(8192)
  })

  it('lmstudio supports native v1 payload shape with models[] and preserves loaded context precedence', async () => {
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({
          models: [{
            id: 'gemma-local',
            max_context_length: 131072,
            loaded_instances: [{ config: { context_length: 73195 } }],
          }],
        }), { status: 200 })
      }
      if (raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'gemma-local' }] }), { status: 200 })
      }
      if (raw.endsWith('/api/v0/models')) return new Response(JSON.stringify({ models: [] }), { status: 200 })
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.models[0]?.maxContextTokens).toBe(73195)
  })

  it('lmstudio falls back to v0 loaded_context_length before any max_context_length', async () => {
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({
          models: [{ id: 'gemma-local', max_context_length: 131072 }],
        }), { status: 200 })
      }
      if (raw.endsWith('/api/v0/models')) {
        return new Response(JSON.stringify({
          models: [{ id: 'gemma-local', loaded_context_length: 73195, max_context_length: 131072 }],
        }), { status: 200 })
      }
      if (raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'gemma-local' }] }), { status: 200 })
      }
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.models[0]?.maxContextTokens).toBe(73195)
  })

  it('lmstudio stops fallback once /api/v1 loaded_instances returns usable context', async () => {
    const seen: string[] = []
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      seen.push(raw)
      if (raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({
          data: [{ id: 'gemma-local', loaded_instances: [{ config: { context_length: 8192 } }], max_context_length: 32768 }],
        }), { status: 200 })
      }
      if (raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'gemma-local' }] }), { status: 200 })
      }
      if (raw.endsWith('/api/v0/models')) {
        return new Response('should-not-be-called', { status: 500 })
      }
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.models[0]?.maxContextTokens).toBe(8192)
    expect(seen.some((u) => u.endsWith('/api/v0/models'))).toBe(false)
  })

  it('lmstudio uses minimal bounded reconciliation (case/trim) only when exact id differs trivially', async () => {
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      if (raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({
          data: [{
            id: '  GEMMA-3-12B-IT  ',
            max_context_length: 131072,
            loaded_instances: [{ config: { context_length: 73728 } }],
          }],
        }), { status: 200 })
      }
      if (raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({
          data: [{ id: 'gemma-3-12b-it', name: 'Gemma 3 12B IT' }],
        }), { status: 200 })
      }
      if (raw.endsWith('/api/v0/models')) return new Response(JSON.stringify({ data: [] }), { status: 200 })
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.models.map((m) => m.id)).toEqual(['gemma-3-12b-it'])
    expect(result.models[0]?.maxContextTokens).toBe(73728)
    expect(result.models[0]?.contextWindow).toBe(73728)
  })

  it('lmstudio stops on first usable source and does not call /api/v0/models when /api/v1/models is usable', async () => {
    const seen: string[] = []
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const raw = typeof url === 'string' ? url : (url instanceof URL ? url.toString() : url.url)
      seen.push(raw)
      if (raw.endsWith('/api/v1/models')) {
        return new Response(JSON.stringify({
          data: [{ id: 'gemma-local', loaded_instances: [{ config: { context_length: 8192 } }] }],
        }), { status: 200 })
      }
      if (raw.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'gemma-local' }] }), { status: 200 })
      }
      if (raw.endsWith('/api/v0/models')) return new Response('unexpected', { status: 500 })
      return new Response('miss', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.models[0]?.maxContextTokens).toBe(8192)
    expect(seen.filter((u) => u.endsWith('/api/v0/models')).length).toBe(0)
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
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/tags')) return new Response('bad', { status: 500 })
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'llama3.2:3b' }] }), { status: 200 })
      }
      if (url.endsWith('/api/ps')) {
        return new Response('bad', { status: 500 })
      }
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        return new Response(JSON.stringify({ model_info: { 'llama.context_length': 8192 } }), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models.map((m) => m.id)).toEqual(['llama3.2:3b'])
    expect(models[0]?.maxContextTokens).toBe(8192)
  })

  it('ollama context precedence prefers /api/ps runtime context over /api/show', async () => {
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/tags')) {
        return new Response(JSON.stringify({ models: [{ name: 'qwen2.5:7b' }] }), { status: 200 })
      }
      if (url.endsWith('/api/ps')) {
        return new Response(JSON.stringify({ models: [{ name: 'qwen2.5:7b', context: 4096 }] }), { status: 200 })
      }
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        return new Response(JSON.stringify({
          parameters: 'num_ctx 8192',
          model_info: { 'llama.context_length': 16384 },
        }), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models[0]?.maxContextTokens).toBe(4096)
  })

  it('ollama reads live-shaped /api/ps context_length and skips /api/show fallback', async () => {
    const seen: string[] = []
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      seen.push(url)
      if (url.endsWith('/api/tags')) {
        return new Response(JSON.stringify({ models: [{ name: 'gemma3:latest' }] }), { status: 200 })
      }
      if (url.endsWith('/api/ps')) {
        return new Response(JSON.stringify({
          models: [
            {
              name: 'gemma3:latest',
              model: 'gemma3:latest',
              size: 123,
              digest: 'abc',
              details: { family: 'gemma' },
              context_length: 73728,
            },
          ],
        }), { status: 200 })
      }
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        return new Response('should-not-be-called', { status: 500 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models[0]?.maxContextTokens).toBe(73728)
    expect(seen.some((u) => u.endsWith('/api/show'))).toBe(false)
  })

  it('ollama stops fallback once /api/ps returns usable runtime context', async () => {
    const seen: string[] = []
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      seen.push(url)
      if (url.endsWith('/api/tags')) {
        return new Response(JSON.stringify({ models: [{ name: 'qwen2.5:7b' }] }), { status: 200 })
      }
      if (url.endsWith('/api/ps')) {
        return new Response(JSON.stringify({ models: [{ name: 'qwen2.5:7b', context: 4096 }] }), { status: 200 })
      }
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        return new Response('should-not-be-called', { status: 500 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models[0]?.maxContextTokens).toBe(4096)
    expect(seen.some((u) => u.endsWith('/api/show'))).toBe(false)
  })

  it('ollama context precedence falls back to /api/show num_ctx before model_info context_length', async () => {
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/tags')) {
        return new Response(JSON.stringify({ models: [{ name: 'qwen2.5:7b' }] }), { status: 200 })
      }
      if (url.endsWith('/api/ps')) return new Response('bad', { status: 500 })
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        return new Response(JSON.stringify({
          parameters: 'temperature 0.8\nnum_ctx 12288\n',
          model_info: { 'llama.context_length': 32768 },
        }), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models[0]?.maxContextTokens).toBe(12288)
  })

  it('ollama runtime context supports exact and :latest equivalence only (no broad reconciliation)', async () => {
    const showBodies: string[] = []
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/tags')) {
        return new Response(JSON.stringify({
          models: [
            { name: 'qwen2.5:latest' },
            { name: 'library/qwen2.5:latest' },
          ],
        }), { status: 200 })
      }
      if (url.endsWith('/api/ps')) {
        return new Response(JSON.stringify({
          models: [
            { model: 'qwen2.5', context: 4096 },
            { model: 'library/qwen2.5', context: 8192 },
          ],
        }), { status: 200 })
      }
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        showBodies.push(String(init.body ?? ''))
        return new Response(JSON.stringify({ parameters: 'num_ctx 16384' }), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models.map((m) => [m.id, m.maxContextTokens])).toEqual([
      ['qwen2.5:latest', 4096],
      ['library/qwen2.5:latest', 8192],
    ])
    expect(showBodies.length).toBe(0)
  })

  it('ollama falls back to /api/show when /api/ps is missing for model', async () => {
    const showBodies: string[] = []
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/tags')) {
        return new Response(JSON.stringify({ models: [{ name: 'library/qwen2.5:latest' }] }), { status: 200 })
      }
      if (url.endsWith('/api/ps')) {
        return new Response(JSON.stringify({ models: [{ model: 'qwen2.5', context: 4096 }] }), { status: 200 })
      }
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        showBodies.push(String(init.body ?? ''))
        return new Response(JSON.stringify({ parameters: 'num_ctx 12288' }), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const models = await Effect.runPromise(discoverOllamaHybrid('http://localhost:11434/v1'))
    expect(models[0]?.id).toBe('library/qwen2.5:latest')
    expect(models[0]?.maxContextTokens).toBe(12288)
    expect(showBodies.length).toBe(1)
  })

  it('llama.cpp enriches context from /props and keeps models when enrichment fails', async () => {
    globalThis.fetch = mock(async (url: string) => {
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'qwen2.5-coder:7b' }] }), { status: 200 })
      }
      if (url.includes('/props?model=')) {
        return new Response(JSON.stringify({ default_generation_settings: { n_ctx: 32768 } }), { status: 200 })
      }
      if (url.endsWith('/props')) return new Response('not found', { status: 404 })
      if (url.endsWith('/slots')) return new Response('not found', { status: 404 })
      return new Response('not found', { status: 404 })
    }) as any

    const provider = getProvider('llama.cpp')
    if (!provider) throw new Error('llama.cpp provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:8080'))

    expect(result.models.map((m) => m.id)).toEqual(['qwen2.5-coder:7b'])
    expect(result.models[0]?.maxContextTokens).toBe(32768)
    expect(result.error).toBeNull()
  })

  it('llama.cpp falls back to /props when /props?model=<id> fails', async () => {
    globalThis.fetch = mock(async (url: string) => {
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'llama3.1:8b' }] }), { status: 200 })
      }
      if (url.includes('/props?model=')) {
        return new Response('boom', { status: 500 })
      }
      if (url.endsWith('/props')) {
        return new Response(JSON.stringify({ default_generation_settings: { n_ctx: 65536 } }), { status: 200 })
      }
      if (url.endsWith('/slots')) return new Response('not found', { status: 404 })
      return new Response('not found', { status: 404 })
    }) as any

    const provider = getProvider('llama.cpp')
    if (!provider) throw new Error('llama.cpp provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:8080'))

    expect(result.error).toBeNull()
    expect(result.status).toBe('success_non_empty')
    expect(result.models.map((m) => m.id)).toEqual(['llama3.1:8b'])
    expect(result.models[0]?.maxContextTokens).toBe(65536)
  })

  it('llama.cpp context precedence falls back to /api/show then /slots then /v1/models meta', async () => {
    const seen: string[] = []
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      seen.push(url)
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'model-a', meta: { n_ctx_train: 32768 } }] }), { status: 200 })
      }
      if (url.includes('/props?model=')) return new Response('bad', { status: 500 })
      if (url.endsWith('/props')) return new Response('bad', { status: 500 })
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        return new Response(JSON.stringify({ model_info: { 'llama.context_length': 8192 } }), { status: 200 })
      }
      if (url.endsWith('/slots')) {
        return new Response(JSON.stringify([{ n_ctx: 16384 }]), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const provider = getProvider('llama.cpp')
    if (!provider) throw new Error('llama.cpp provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:8080'))

    expect(result.models[0]?.maxContextTokens).toBe(8192)
    expect(seen.some((u) => u.endsWith('/slots'))).toBe(false)
  })

  it('llama.cpp falls back to /slots when /props and /api/show are unavailable', async () => {
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'model-b', meta: { n_ctx_train: 32768 } }] }), { status: 200 })
      }
      if (url.includes('/props?model=')) return new Response('bad', { status: 500 })
      if (url.endsWith('/props')) return new Response('bad', { status: 500 })
      if (url.endsWith('/api/show') && init?.method === 'POST') return new Response('bad', { status: 500 })
      if (url.endsWith('/slots')) {
        return new Response(JSON.stringify([{ n_ctx: 16384 }]), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const provider = getProvider('llama.cpp')
    if (!provider) throw new Error('llama.cpp provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:8080'))

    expect(result.models[0]?.maxContextTokens).toBe(16384)
  })

  it('llama.cpp treats props zero n_ctx as missing and falls back to /api/show', async () => {
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'model-z', meta: { n_ctx_train: 32768 } }] }), { status: 200 })
      }
      if (url.includes('/props?model=')) {
        return new Response(JSON.stringify({ default_generation_settings: { n_ctx: 0 } }), { status: 200 })
      }
      if (url.endsWith('/props')) {
        return new Response(JSON.stringify({ default_generation_settings: { n_ctx: 0 } }), { status: 200 })
      }
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        return new Response(JSON.stringify({ model_info: { 'llama.context_length': 6144 } }), { status: 200 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const provider = getProvider('llama.cpp')
    if (!provider) throw new Error('llama.cpp provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:8080'))

    expect(result.models[0]?.maxContextTokens).toBe(6144)
  })

  it('llama.cpp falls back to /v1/models meta.n_ctx_train when higher-precedence sources are unavailable', async () => {
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'model-c', meta: { n_ctx_train: 24576 } }] }), { status: 200 })
      }
      if (url.includes('/props?model=')) return new Response('bad', { status: 500 })
      if (url.endsWith('/props')) return new Response('bad', { status: 500 })
      if (url.endsWith('/api/show') && init?.method === 'POST') return new Response('bad', { status: 500 })
      if (url.endsWith('/slots')) return new Response('bad', { status: 500 })
      return new Response('not found', { status: 404 })
    }) as any

    const provider = getProvider('llama.cpp')
    if (!provider) throw new Error('llama.cpp provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:8080'))

    expect(result.models[0]?.maxContextTokens).toBe(24576)
  })

  it('lmstudio keeps discovered model available when context enrichment fails', async () => {
    globalThis.fetch = mock(async (url: string) => {
      if (url.endsWith('/v1/models')) {
        return new Response(JSON.stringify({ data: [{ id: 'gemma-3-4b' }] }), { status: 200 })
      }
      if (url.endsWith('/api/v1/models')) return new Response('bad', { status: 500 })
      if (url.endsWith('/api/v0/models')) return new Response('bad', { status: 500 })
      return new Response('not found', { status: 404 })
    }) as any

    const provider = getProvider('lmstudio')
    if (!provider) throw new Error('lmstudio provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:1234/v1'))

    expect(result.error).toBeNull()
    expect(result.status).toBe('success_non_empty')
    expect(result.models.map((m) => m.id)).toEqual(['gemma-3-4b'])
    expect(result.models[0]?.maxContextTokens).toBeNull()
  })

  it('ollama keeps discovered model available when /api/show enrichment fails', async () => {
    globalThis.fetch = mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/tags')) {
        return new Response(JSON.stringify({ models: [{ name: 'qwen2.5:7b' }] }), { status: 200 })
      }
      if (url.endsWith('/api/show') && init?.method === 'POST') {
        return new Response('bad', { status: 500 })
      }
      if (url.endsWith('/v1/models')) {
        return new Response('unexpected', { status: 500 })
      }
      return new Response('not found', { status: 404 })
    }) as any

    const provider = getProvider('ollama')
    if (!provider) throw new Error('ollama provider not found')
    const result = await Effect.runPromise(discoverLocalProviderModels(provider, 'http://localhost:11434/v1'))

    expect(result.error).toBeNull()
    expect(result.status).toBe('success_non_empty')
    expect(result.models.map((m) => m.id)).toEqual(['qwen2.5:7b'])
    expect(result.models[0]?.maxContextTokens).toBeNull()
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
