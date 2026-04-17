import { useState, useCallback } from 'react'
import {
  startAnthropicOAuth,
  exchangeAnthropicCode,
  startOpenAIBrowserOAuth,
  startOpenAIDeviceOAuth,
  getProvider,
  type ProviderDefinition,
} from '@magnitudedev/agent'
import { writeTextToClipboard } from '../utils/clipboard'
import { trackProviderConnected } from '@magnitudedev/telemetry'
import { useProviderRuntime } from '../providers/provider-runtime'
import { useStorage } from '../providers/storage-provider'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface OAuthState {
  provider: ProviderDefinition
  mode: 'paste' | 'auto'
  methodLabel: string
  url: string
  codeVerifier?: string
  codeError?: string | null
  isSubmitting?: boolean
  userCode?: string
  cleanup?: () => void
}

export interface ApiKeySetupState {
  provider: ProviderDefinition
  envKeyHint: string
  existingKey?: string
}

export interface UseAuthFlowOptions {
  onAuthSuccess: (providerId: string, providerName: string) => void
  onAuthCancel: () => void
  onMessage: (message: string) => void
  showEphemeral: (msg: string, color: string, duration?: number) => void
  theme: { success: string; error: string }
  reloadProviderState?: () => Promise<void>
}

export interface EndpointSetupState {
  provider: ProviderDefinition
  initialOptions?: { baseUrl?: string; modelId?: string } | null
}

export interface UseAuthFlowReturn {
  oauthState: OAuthState | null
  apiKeySetup: ApiKeySetupState | null
  endpointSetup: EndpointSetupState | null
  showAuthMethodOverlay: boolean
  authMethodProvider: ProviderDefinition | null
  startAuthForProvider: (provider: ProviderDefinition, methodIndex: number) => Promise<void>
  openApiKeyOverlay: (provider: ProviderDefinition, envKeyHint: string, existingKey?: string) => void
  openAuthMethodPicker: (provider: ProviderDefinition) => void
  closeAuthMethodPicker: () => void
  handleOAuthCodeSubmit: (code: string) => Promise<void>
  handleOAuthCancel: () => void
  handleOAuthCopyCode: () => Promise<void>
  handleOAuthCopyUrl: () => Promise<void>
  handleApiKeySubmit: (key: string) => Promise<void>
  handleApiKeyCancel: () => void
  handleEndpointSetupSubmit: (config: { providerId: string; url: string; modelId?: string; apiKey?: string }) => Promise<void>
  handleEndpointSetupCancel: () => void
  cancelAll: () => void
}

export function useAuthFlow({
  onAuthSuccess,
  onAuthCancel,
  onMessage,
  showEphemeral,
  theme,
  reloadProviderState,
}: UseAuthFlowOptions): UseAuthFlowReturn {
  const runtime = useProviderRuntime()
  const storage = useStorage()
  const [oauthState, setOauthState] = useState<OAuthState | null>(null)
  const [apiKeySetup, setApiKeySetup] = useState<ApiKeySetupState | null>(null)
  const [endpointSetup, setEndpointSetup] = useState<EndpointSetupState | null>(null)
  const [showAuthMethodOverlay, setShowAuthMethodOverlay] = useState(false)
  const [authMethodProvider, setAuthMethodProvider] = useState<ProviderDefinition | null>(null)

  const reload = useCallback(async () => {
    await reloadProviderState?.()
  }, [reloadProviderState])

  const openApiKeyOverlay = useCallback((provider: ProviderDefinition, envKeyHint: string, existingKey?: string) => {
    setApiKeySetup({ provider, envKeyHint, existingKey })
  }, [])

  const openAuthMethodPicker = useCallback((provider: ProviderDefinition) => {
    setShowAuthMethodOverlay(true)
    setAuthMethodProvider(provider)
  }, [])

  const closeAuthMethodPicker = useCallback(() => {
    setShowAuthMethodOverlay(false)
    setAuthMethodProvider(null)
    onAuthCancel()
  }, [onAuthCancel])

  const startAuthForProvider = useCallback(async (provider: ProviderDefinition, methodIndex: number) => {
    const method = provider.authMethods[methodIndex]
    setShowAuthMethodOverlay(false)
    setAuthMethodProvider(null)

    if (method.type === 'api-key') {
      const envKey = method.envKeys?.find(k => process.env[k])
      if (envKey) {
        showEphemeral(`Connected ${provider.name} (${envKey})`, theme.success)
        await reload()
        onAuthSuccess(provider.id, provider.name)
      } else {
        const storedAuth = await storage.auth.get(provider.id)
        const existingKey = storedAuth?.type === 'api' ? storedAuth.key : undefined
        setApiKeySetup({ provider, envKeyHint: method.envKeys?.[0] ?? '', existingKey })
      }
    } else if (method.type === 'oauth-pkce') {
      const { authUrl, codeVerifier } = startAnthropicOAuth()
      setOauthState({
        provider, mode: 'paste', methodLabel: method.label,
        url: authUrl, codeVerifier, codeError: null, isSubmitting: false,
      })
    } else if (method.type === 'oauth-device') {
      try {
        const result = await startOpenAIDeviceOAuth()
        setOauthState({
          provider, mode: 'auto', methodLabel: method.label,
          url: result.verificationUrl, userCode: result.userCode,
        })
        result.poll().then(async auth => {
          await storage.auth.set(provider.id, { ...auth, oauthMethod: 'oauth-device' })
          trackProviderConnected({ providerId: provider.id, authType: auth.type })
          await reload()
          setOauthState(null)
          showEphemeral(`Connected ${provider.name}`, theme.success)
          onAuthSuccess(provider.id, provider.name)
        }).catch(err => {
          setOauthState(null)
          showEphemeral(`Auth failed: ${err.message}`, theme.error, 8000)
        })
      } catch (err: any) {
        showEphemeral(`Failed to start auth: ${err.message}`, theme.error, 8000)
      }
    } else if (method.type === 'oauth-browser') {
      try {
        const result = await startOpenAIBrowserOAuth()
        setOauthState({
          provider, mode: 'auto', methodLabel: method.label,
          url: result.authUrl, cleanup: result.stop,
        })
        result.waitForCallback().then(async auth => {
          await storage.auth.set(provider.id, { ...auth, oauthMethod: 'oauth-browser' })
          trackProviderConnected({ providerId: provider.id, authType: auth.type })
          await reload()
          setOauthState(null)
          showEphemeral(`Connected ${provider.name}`, theme.success)
          onAuthSuccess(provider.id, provider.name)
        }).catch(err => {
          setOauthState(null)
          showEphemeral(`Auth failed: ${err.message}`, theme.error, 8000)
        })
      } catch (err: any) {
        showEphemeral(`Failed to start auth: ${err.message}`, theme.error, 8000)
      }
    } else if (method.type === 'none') {
      const existing = await storage.config.getProviderOptions(provider.id)
      setEndpointSetup({
        provider,
        initialOptions: {
          baseUrl: existing?.baseUrl,
          modelId: typeof existing?.modelId === 'string' ? existing.modelId : undefined,
        },
      })
    }
  }, [showEphemeral, theme.success, theme.error, onAuthSuccess, onMessage, storage, reload])

  const handleOAuthCodeSubmit = useCallback(async (code: string) => {
    if (!oauthState || oauthState.mode !== 'paste' || !oauthState.codeVerifier) return
    setOauthState(prev => prev ? { ...prev, isSubmitting: true, codeError: null } : null)
    try {
      const auth = await exchangeAnthropicCode(code, oauthState.codeVerifier)
      await storage.auth.set(oauthState.provider.id, { ...auth, oauthMethod: 'oauth-pkce' })
      trackProviderConnected({ providerId: oauthState.provider.id, authType: auth.type })
      await reload()
      const providerName = oauthState.provider.name
      const providerId = oauthState.provider.id
      setOauthState(null)
      showEphemeral(`Connected ${providerName}`, theme.success)
      onAuthSuccess(providerId, providerName)
    } catch {
      setOauthState(prev => prev ? { ...prev, isSubmitting: false, codeError: 'Invalid code' } : null)
    }
  }, [oauthState, showEphemeral, theme.success, onAuthSuccess, storage, reload])

  const handleOAuthCancel = useCallback(() => {
    oauthState?.cleanup?.()
    setOauthState(null)
    onAuthCancel()
  }, [oauthState, onAuthCancel])

  const handleOAuthCopyCode = useCallback(async () => {
    if (!oauthState?.userCode) return
    await writeTextToClipboard(oauthState.userCode)
  }, [oauthState])

  const handleOAuthCopyUrl = useCallback(async () => {
    if (!oauthState?.url) return
    await writeTextToClipboard(oauthState.url)
  }, [oauthState])

  const handleEndpointSetupSubmit = useCallback(async (config: { providerId: string; url: string; modelId?: string; apiKey?: string }) => {
    const provider = getProvider(config.providerId)
    setEndpointSetup(null)

    if (config.apiKey) {
      await storage.auth.set(config.providerId, { type: 'api' as const, key: config.apiKey })
      trackProviderConnected({ providerId: config.providerId, authType: 'api' })
    } else {
      trackProviderConnected({ providerId: config.providerId, authType: 'none' })
    }

    await storage.config.updateFull((current) => {
      const existing = current.providers?.[config.providerId] ?? {}
      const rememberedRaw = (existing as any).rememberedModelIds
      const remembered = Array.isArray(rememberedRaw) ? rememberedRaw.filter((id): id is string => typeof id === 'string') : []
      return {
        ...current,
        providers: {
          ...(current.providers ?? {}),
          [config.providerId]: {
            ...existing,
            baseUrl: config.url,
            ...(config.modelId ? { modelId: config.modelId } : {}),
            ...(config.modelId ? { rememberedModelIds: Array.from(new Set([...remembered, config.modelId])) } : {}),
          },
        },
      }
    })

    await runtime.catalog.refresh()
    await reload()
    const providerName = provider?.name ?? config.providerId
    showEphemeral(`Connected ${providerName}`, theme.success)
    onAuthSuccess(config.providerId, providerName)
  }, [showEphemeral, theme.success, onAuthSuccess, storage, runtime, reload])

  const handleEndpointSetupCancel = useCallback(() => {
    setEndpointSetup(null)
    onAuthCancel()
  }, [onAuthCancel])

  const handleApiKeySubmit = useCallback(async (key: string) => {
    if (!apiKeySetup) return
    const auth = { type: 'api' as const, key }
    await storage.auth.set(apiKeySetup.provider.id, auth)
    trackProviderConnected({ providerId: apiKeySetup.provider.id, authType: 'api' })
    await reload()
    const providerName = apiKeySetup.provider.name
    const providerId = apiKeySetup.provider.id
    setApiKeySetup(null)
    showEphemeral(`Connected ${providerName} (API Key)`, theme.success)
    onAuthSuccess(providerId, providerName)
  }, [apiKeySetup, showEphemeral, theme.success, onAuthSuccess, storage, reload])

  const handleApiKeyCancel = useCallback(() => {
    setApiKeySetup(null)
    onAuthCancel()
  }, [onAuthCancel])

  const cancelAll = useCallback(() => {
    oauthState?.cleanup?.()
    setOauthState(null)
    setApiKeySetup(null)
    setEndpointSetup(null)
    setShowAuthMethodOverlay(false)
    setAuthMethodProvider(null)
  }, [oauthState])

  return {
    oauthState,
    apiKeySetup,
    endpointSetup,
    showAuthMethodOverlay,
    authMethodProvider,
    startAuthForProvider,
    openApiKeyOverlay,
    openAuthMethodPicker,
    closeAuthMethodPicker,
    handleOAuthCodeSubmit,
    handleOAuthCancel,
    handleOAuthCopyCode,
    handleOAuthCopyUrl,
    handleApiKeySubmit,
    handleApiKeyCancel,
    handleEndpointSetupSubmit,
    handleEndpointSetupCancel,
    cancelAll,
  }
}