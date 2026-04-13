import { afterEach, describe, expect, it, vi } from 'vitest'
import { detectProviderAuthMethods, detectProviders } from '../detect'
import type { AuthInfo, ProviderOptions } from '../types'

describe('detect local-family providers', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    delete process.env.FIREWORKS_API_KEY
    delete process.env.OPENAI_API_KEY
  })

  // First-class locals (lmstudio, ollama, llama.cpp) use discovery status
  it('marks first-class local connected when lastDiscoveryStatus is success_non_empty', () => {
    const storedAuth: Record<string, AuthInfo> = {}
    const providerOptionsById: Record<string, ProviderOptions | undefined> = {
      lmstudio: { lastDiscoveryStatus: 'success_non_empty' },
      ollama: { lastDiscoveryStatus: 'success_empty' },
    }

    const detected = detectProviders(storedAuth, providerOptionsById)
    const ids = new Set(detected.map((d) => d.provider.id))

    expect(ids.has('lmstudio')).toBe(true)
    expect(ids.has('ollama')).toBe(false)
  })

  it('does not mark first-class local as connected when discovery failed', () => {
    const detected = detectProviders({}, {
      lmstudio: { lastDiscoveryStatus: 'failure' },
    })
    expect(detected.some((d) => d.provider.id === 'lmstudio')).toBe(false)
  })

  it('does not mark first-class local as connected when no discovery has run', () => {
    const detected = detectProviders({}, { lmstudio: {} })
    expect(detected.some((d) => d.provider.id === 'lmstudio')).toBe(false)
  })

  // DIY local (openai-compatible-local) uses baseUrl presence
  it('marks DIY local connected when baseUrl is set', () => {
    const detected = detectProviders({}, {
      'openai-compatible-local': { baseUrl: 'http://localhost:9000/v1' },
    })
    expect(detected.some((d) => d.provider.id === 'openai-compatible-local')).toBe(true)
  })

  it('does not mark DIY local connected with whitespace-only baseUrl', () => {
    const detected = detectProviders({}, {
      'openai-compatible-local': { baseUrl: '   ' },
    })
    expect(detected.some((d) => d.provider.id === 'openai-compatible-local')).toBe(false)
  })

  // detectProviderAuthMethods for first-class locals
  it('returns none-auth method as connected for first-class local with discovered models', () => {
    const status = detectProviderAuthMethods(
      'llama.cpp',
      {},
      { 'llama.cpp': { lastDiscoveryStatus: 'success_non_empty' } },
    )

    expect(status).not.toBeNull()
    expect(status!.anyConnected).toBe(true)
    expect(status!.methods.some((m) => m.method.type === 'none' && m.connected)).toBe(true)
  })

  it('does not connect none-auth method for first-class local with empty discovery', () => {
    const status = detectProviderAuthMethods(
      'llama.cpp',
      {},
      { 'llama.cpp': { lastDiscoveryStatus: 'success_empty' } },
    )

    expect(status).not.toBeNull()
    expect(status!.anyConnected).toBe(false)
  })

  it('detects Fireworks from FIREWORKS_API_KEY env auth', () => {
    process.env.FIREWORKS_API_KEY = 'fireworks-env-key'

    const detected = detectProviders({}, {})
    const fireworks = detected.find((d) => d.provider.id === 'fireworks-ai')

    expect(fireworks).toBeDefined()
    expect(fireworks?.source).toBe('env')
    expect(fireworks?.auth).toEqual({ type: 'api', key: 'fireworks-env-key' })
  })

  it('reports Fireworks api-key auth method as connected from env', () => {
    process.env.FIREWORKS_API_KEY = 'fireworks-env-key'

    const status = detectProviderAuthMethods('fireworks-ai', {}, {})

    expect(status).not.toBeNull()
    expect(status!.anyConnected).toBe(true)
    expect(status!.methods).toEqual([
      {
        method: { type: 'api-key', label: 'API key', envKeys: ['FIREWORKS_API_KEY'] },
        methodIndex: 0,
        connected: true,
        source: 'env',
        auth: { type: 'api', key: 'fireworks-env-key' },
      },
    ])
  })

  it('does not detect Fireworks from OPENAI_API_KEY alone', () => {
    process.env.OPENAI_API_KEY = 'openai-only-key'

    const detected = detectProviders({}, {})
    expect(detected.some((d) => d.provider.id === 'fireworks-ai')).toBe(false)

    const status = detectProviderAuthMethods('fireworks-ai', {}, {})
    expect(status).not.toBeNull()
    expect(status!.anyConnected).toBe(false)
  })
})
