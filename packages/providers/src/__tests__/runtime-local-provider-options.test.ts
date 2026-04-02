import { describe, expect, it } from 'bun:test'
import { detectProviderAuthMethods, detectProviders } from '../detect'

describe('provider runtime local-family detection uses config.providers', () => {
  it('reflects local-family connectivity from provider options map', () => {
    const storedAuth = {}
    const providerOptionsById = {
      lmstudio: { baseUrl: 'http://localhost:1234/v1' },
    }

    const detected = detectProviders(storedAuth, providerOptionsById)
    expect(detected.some((d) => d.provider.id === 'lmstudio')).toBe(true)

    const status = detectProviderAuthMethods('lmstudio', storedAuth, providerOptionsById)
    expect(status?.anyConnected).toBe(true)
    expect(status?.methods.some((m) => m.method.type === 'none' && m.connected)).toBe(true)
  })
})
