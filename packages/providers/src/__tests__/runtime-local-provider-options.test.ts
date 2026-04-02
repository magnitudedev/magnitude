import { describe, expect, it } from 'bun:test'
import { detectProviderAuthMethods, detectProviders } from '../detect'

describe('provider runtime local-family detection uses config.providers', () => {
  it('reflects first-class local connectivity from discovery status', () => {
    const storedAuth = {}
    const providerOptionsById = {
      lmstudio: { lastDiscoveryStatus: 'success_non_empty' as const },
    }

    const detected = detectProviders(storedAuth, providerOptionsById)
    expect(detected.some((d) => d.provider.id === 'lmstudio')).toBe(true)

    const status = detectProviderAuthMethods('lmstudio', storedAuth, providerOptionsById)
    expect(status?.anyConnected).toBe(true)
    expect(status?.methods.some((m) => m.method.type === 'none' && m.connected)).toBe(true)
  })
})
