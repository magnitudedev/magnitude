import type { AuthInfo, ProviderDefinition } from '../types'

export function resolveEnvAuth(provider: ProviderDefinition): AuthInfo | null {
  for (const method of provider.authMethods) {
    if (method.type !== 'api-key' || !method.envKeys) continue

    for (const envKey of method.envKeys) {
      const value = process.env[envKey]
      if (value) {
        return { type: 'api', key: value }
      }
    }
  }

  return null
}

export function resolveNonStoredAuth(provider: ProviderDefinition): AuthInfo | null {
  return resolveEnvAuth(provider)
}
