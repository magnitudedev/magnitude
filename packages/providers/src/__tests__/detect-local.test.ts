import { describe, expect, it } from 'bun:test'
import { detectProviderAuthMethods, detectProviders } from '../detect'
import type { AuthInfo, ProviderOptions } from '../types'

describe('detect local-family providers', () => {
  it('marks local-family providers connected only when baseUrl is non-empty after trim', () => {
    const storedAuth: Record<string, AuthInfo> = {}
    const providerOptionsById: Record<string, ProviderOptions | undefined> = {
      lmstudio: { baseUrl: 'http://localhost:1234/v1' },
      ollama: { baseUrl: '   ' },
    }

    const detected = detectProviders(storedAuth, providerOptionsById)
    const ids = new Set(detected.map((d) => d.provider.id))

    expect(ids.has('lmstudio')).toBe(true)
    expect(ids.has('ollama')).toBe(false)
  })

  it('returns none-auth method as connected for local provider with baseUrl', () => {
    const status = detectProviderAuthMethods(
      'llama.cpp',
      {},
      { 'llama.cpp': { baseUrl: 'http://localhost:8080' } },
    )

    expect(status).not.toBeNull()
    expect(status!.anyConnected).toBe(true)
    expect(status!.methods.some((m) => m.method.type === 'none' && m.connected)).toBe(true)
  })

  it('does not connect none-auth method for whitespace-only baseUrl', () => {
    const status = detectProviderAuthMethods(
      'llama.cpp',
      {},
      { 'llama.cpp': { baseUrl: '   ' } },
    )

    expect(status).not.toBeNull()
    expect(status!.anyConnected).toBe(false)
  })
})
