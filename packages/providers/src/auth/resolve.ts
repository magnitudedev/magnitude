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

export function resolveAwsAuth(provider: ProviderDefinition): AuthInfo | null {
  if (!provider.authMethods.some((method) => method.type === 'aws-chain')) {
    return null
  }

  const hasAccessKey = !!process.env.AWS_ACCESS_KEY_ID
  const hasProfile = !!process.env.AWS_PROFILE

  if (hasAccessKey || hasProfile) {
    return {
      type: 'aws',
      profile: process.env.AWS_PROFILE,
      region: process.env.AWS_DEFAULT_REGION || process.env.AWS_REGION,
    }
  }

  return null
}

export function resolveGcpAuth(provider: ProviderDefinition): AuthInfo | null {
  if (!provider.authMethods.some((method) => method.type === 'gcp-credentials')) {
    return null
  }

  const credentialsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS
  if (!credentialsPath) {
    return null
  }

  return {
    type: 'gcp',
    credentialsPath,
    project: process.env.GCLOUD_PROJECT || process.env.GOOGLE_CLOUD_PROJECT,
    location: process.env.GOOGLE_CLOUD_LOCATION,
  }
}

export function resolveNonStoredAuth(provider: ProviderDefinition): AuthInfo | null {
  return (
    resolveEnvAuth(provider)
    ?? resolveAwsAuth(provider)
    ?? resolveGcpAuth(provider)
  )
}
