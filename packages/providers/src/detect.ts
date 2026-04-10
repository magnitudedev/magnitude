/**
 * Auto-detect available providers from environment variables and stored auth.
 */

import { PROVIDERS } from './registry'

import type { AuthInfo, AuthMethodDef, ProviderDefinition, ProviderOptions } from './types'

export interface DetectedProvider {
  provider: ProviderDefinition
  auth: AuthInfo | null
  source: 'stored' | 'env' | 'none'  // How auth was found
}

/**
 * Scan all providers and return those that have available auth.
 * Checks stored auth first, then env vars.
 */
export function detectProviders(
  storedAuth: Record<string, AuthInfo>,
  providerOptionsById?: Record<string, ProviderOptions | undefined>,
): DetectedProvider[] {
  const detected: DetectedProvider[] = []

  for (const provider of PROVIDERS) {
    // 1. Check stored auth (OAuth tokens, manually entered keys, etc.)
    const stored = storedAuth[provider.id]
    if (stored) {
      detected.push({ provider, auth: stored, source: 'stored' })
      continue
    }

    // 2. Check env vars for API key auth methods
    const envAuth = checkEnvAuth(provider)
    if (envAuth) {
      detected.push({ provider, auth: envAuth, source: 'env' })
      continue
    }

    // 3. Check for auth-less providers (e.g., local runtimes)
    if (provider.authMethods.some(m => m.type === 'none')) {
      if (provider.providerFamily === 'local') {
        const options = providerOptionsById?.[provider.id]
        const connected = provider.defaultBaseUrl
          ? options?.lastDiscoveryStatus === 'success_non_empty'
          : !!options?.baseUrl?.trim()
        if (connected) {
          detected.push({ provider, auth: null, source: 'none' })
        }
      } else {
        detected.push({ provider, auth: null, source: 'none' })
      }
      continue
    }

    // 4. Check AWS credential chain (profile-based, no explicit key needed)
    if (provider.authMethods.some(m => m.type === 'aws-chain')) {
      const awsAuth = checkAwsEnv(provider)
      if (awsAuth) {
        detected.push({ provider, auth: awsAuth, source: 'env' })
        continue
      }
    }

    // 5. Check GCP credentials
    if (provider.authMethods.some(m => m.type === 'gcp-credentials')) {
      const gcpAuth = checkGcpEnv(provider)
      if (gcpAuth) {
        detected.push({ provider, auth: gcpAuth, source: 'env' })
        continue
      }
    }
  }

  return detected
}

/**
 * Get the best available provider (first detected, preferring stored auth).
 */
export function detectDefaultProvider(
  storedAuth: Record<string, AuthInfo>,
  providerOptionsById?: Record<string, ProviderOptions | undefined>,
): DetectedProvider | null {
  const all = detectProviders(storedAuth, providerOptionsById)
  // Prefer stored auth (user explicitly configured)
  const stored = all.find(d => d.source === 'stored')
  if (stored) return stored
  // Then env-based
  const env = all.find(d => d.source === 'env')
  if (env) return env
  // Then auth-less
  return all[0] ?? null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function checkEnvAuth(provider: ProviderDefinition): AuthInfo | null {
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

function checkAwsEnv(provider: ProviderDefinition): AuthInfo | null {
  // AWS credential chain: explicit keys, profile, or instance metadata
  const hasAccessKey = !!process.env.AWS_ACCESS_KEY_ID
  const hasProfile = !!process.env.AWS_PROFILE
  const hasRegion = !!(process.env.AWS_DEFAULT_REGION || process.env.AWS_REGION)

  if (hasAccessKey || hasProfile) {
    return {
      type: 'aws',
      profile: process.env.AWS_PROFILE,
      region: process.env.AWS_DEFAULT_REGION || process.env.AWS_REGION,
    }
  }
  return null
}

function checkGcpEnv(provider: ProviderDefinition): AuthInfo | null {
  const credPath = process.env.GOOGLE_APPLICATION_CREDENTIALS
  if (credPath) {
    return {
      type: 'gcp',
      credentialsPath: credPath,
      project: process.env.GCLOUD_PROJECT || process.env.GOOGLE_CLOUD_PROJECT,
      location: process.env.GOOGLE_CLOUD_LOCATION,
    }
  }
  return null
}

// ---------------------------------------------------------------------------
// Per-method auth detection
// ---------------------------------------------------------------------------

export interface DetectedAuthMethod {
  method: AuthMethodDef
  methodIndex: number
  connected: boolean
  source: 'stored' | 'env' | 'none' | null
  auth: AuthInfo | null
}

export interface ProviderAuthMethodStatus {
  provider: ProviderDefinition
  methods: DetectedAuthMethod[]
  anyConnected: boolean
}

/**
 * Detect auth status for each auth method of a provider independently.
 * Unlike detectProviders() which returns one entry per provider,
 * this returns per-method status so the UI can show all methods.
 */
export function detectProviderAuthMethods(
  providerId: string,
  storedAuth: Record<string, AuthInfo>,
  providerOptionsById?: Record<string, ProviderOptions | undefined>,
): ProviderAuthMethodStatus | null {
  const provider = PROVIDERS.find(p => p.id === providerId)
  if (!provider) return null

  const stored = storedAuth[providerId] ?? null
  const methods: DetectedAuthMethod[] = []

  for (let i = 0; i < provider.authMethods.length; i++) {
    const method = provider.authMethods[i]
    let connected = false
    let source: DetectedAuthMethod['source'] = null
    let auth: AuthInfo | null = null

    if (method.type === 'api-key') {
      // Check env vars first (independent of stored auth type)
      const envKey = method.envKeys?.find(k => process.env[k])
      if (envKey) {
        connected = true
        source = 'env'
        auth = { type: 'api', key: process.env[envKey]! }
      }
      // Stored API key overrides env in display priority
      if (stored?.type === 'api') {
        connected = true
        source = 'stored'
        auth = stored
      }
    } else if (method.type === 'oauth-pkce' || method.type === 'oauth-device' || method.type === 'oauth-browser') {
      if (stored?.type === 'oauth') {
        connected = true
        source = 'stored'
        auth = stored
      }
    } else if (method.type === 'aws-chain') {
      if (stored?.type === 'aws') {
        connected = true
        source = 'stored'
        auth = stored
      } else {
        const awsAuth = checkAwsEnv(provider)
        if (awsAuth) {
          connected = true
          source = 'env'
          auth = awsAuth
        }
      }
    } else if (method.type === 'gcp-credentials') {
      if (stored?.type === 'gcp') {
        connected = true
        source = 'stored'
        auth = stored
      } else {
        const gcpAuth = checkGcpEnv(provider)
        if (gcpAuth) {
          connected = true
          source = 'env'
          auth = gcpAuth
        }
      }
    } else if (method.type === 'none') {
      if (provider.providerFamily === 'local') {
        const options = providerOptionsById?.[provider.id]
        const isConnected = provider.defaultBaseUrl
          ? options?.lastDiscoveryStatus === 'success_non_empty'
          : !!options?.baseUrl?.trim()
        if (isConnected) {
          connected = true
          source = 'none'
        }
      } else {
        connected = true
        source = 'none'
      }
    }

    methods.push({ method, methodIndex: i, connected, source, auth })
  }

  return {
    provider,
    methods,
    anyConnected: methods.some(m => m.connected),
  }
}
