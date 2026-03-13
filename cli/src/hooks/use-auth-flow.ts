import { useState, useCallback } from 'react'
import {
  startAnthropicOAuth,
  exchangeAnthropicCode,
  startOpenAIBrowserOAuth,
  startOpenAIDeviceOAuth,
  startCopilotAuth,
  type ProviderDefinition,
} from '@magnitudedev/agent'
import { writeTextToClipboard } from '../utils/clipboard'
import { trackProviderConnected } from '@magnitudedev/telemetry'
import { useProviderRuntime } from '../providers/provider-runtime'

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

export interface UseAuthFlowReturn {
  oauthState: OAuthState | null
  apiKeySetup: ApiKeySetupState | null
  showLocalSetup: boolean
  showAuthMethodOverlay: boolean
  authMethodProvider: ProviderDefinition | null
  startAuthForProvider: (provider: ProviderDefinition, methodIndex: number) => Promise<void>
  openApiKeyOverlay: (provider: ProviderDefinition, envKeyHint: string, existingKey?: string) => void
  openAuthMethodPicker: (provider: ProviderDefinition) => void
  closeAuthMethodPicker: () => void
  handleOAuthCodeSubmit: (code: string) => Promise<void>
  handleOAuthCancel: () => void
  handleOAuthCopyCode: () => void
  handleOAuthCopyUrl: () => void
  handleApiKeySubmit: (key: string) => Promise<void>
  handleApiKeyCancel: () => void
  handleLocalSetupSubmit: (config: { url: string; modelId: string; apiKey?: string }) => Promise<void>
  handleLocalSetupCancel: () => void
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
  const [oauthState, setOauthState] = useState<OAuthState | null>(null)
  const [apiKeySetup, setApiKeySetup] = useState<ApiKeySetupState | null>(null)
  const [showLocalSetup, setShowLocalSetup] = useState(false)
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
        const storedAuth = await runtime.auth.getAuth(provider.id)
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
      const startFn = provider.id === 'github-copilot' ? startCopilotAuth : startOpenAIDeviceOAuth
      try {
        const result = await startFn()
        setOauthState({
          provider, mode: 'auto', methodLabel: method.label,
          url: result.verificationUrl, userCode: result.userCode,
        })
        result.poll().then(async auth => {
          await runtime.auth.setAuth(provider.id, auth)
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
          await runtime.auth.setAuth(provider.id, auth)
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
    } else if (method.type === 'aws-chain') {
      const hasAccessKey = !!process.env.AWS_ACCESS_KEY_ID
      const hasProfile = !!process.env.AWS_PROFILE
      if (hasAccessKey || hasProfile) {
        const auth = {
          type: 'aws' as const,
          profile: process.env.AWS_PROFILE,
          region: process.env.AWS_DEFAULT_REGION || process.env.AWS_REGION,
        }
        await runtime.auth.setAuth(provider.id, auth)
        trackProviderConnected({ providerId: provider.id, authType: 'aws' })
        await reload()
        showEphemeral(`Connected ${provider.name}`, theme.success)
        onAuthSuccess(provider.id, provider.name)
      } else {
        onMessage('Configure AWS credentials using aws configure, or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables, then restart Magnitude.')
      }
    } else if (method.type === 'gcp-credentials') {
      const credPath = process.env.GOOGLE_APPLICATION_CREDENTIALS
      if (credPath) {
        const auth = {
          type: 'gcp' as const,
          credentialsPath: credPath,
          project: process.env.GCLOUD_PROJECT || process.env.GOOGLE_CLOUD_PROJECT,
          location: process.env.GOOGLE_CLOUD_LOCATION,
        }
        await runtime.auth.setAuth(provider.id, auth)
        trackProviderConnected({ providerId: provider.id, authType: 'gcp' })
        await reload()
        showEphemeral(`Connected ${provider.name}`, theme.success)
        onAuthSuccess(provider.id, provider.name)
      } else {
        onMessage('Run gcloud auth application-default login, or set GOOGLE_APPLICATION_CREDENTIALS to your service account key file path, then restart Magnitude.')
      }
    } else if (method.type === 'none') {
      if (provider.id === 'local') {
        setShowLocalSetup(true)
      } else {
        await reload()
        showEphemeral(`Connected ${provider.name}`, theme.success)
        onAuthSuccess(provider.id, provider.name)
      }
    }
  }, [showEphemeral, theme.success, theme.error, onAuthSuccess, onMessage, runtime, reload])

  const handleOAuthCodeSubmit = useCallback(async (code: string) => {
    if (!oauthState || oauthState.mode !== 'paste' || !oauthState.codeVerifier) return
    setOauthState(prev => prev ? { ...prev, isSubmitting: true, codeError: null } : null)
    try {
      const auth = await exchangeAnthropicCode(code, oauthState.codeVerifier)
      await runtime.auth.setAuth(oauthState.provider.id, auth)
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
  }, [oauthState, showEphemeral, theme.success, onAuthSuccess, runtime, reload])

  const handleOAuthCancel = useCallback(() => {
    oauthState?.cleanup?.()
    setOauthState(null)
    onAuthCancel()
  }, [oauthState, onAuthCancel])

  const handleOAuthCopyCode = useCallback(() => {
    if (!oauthState?.userCode) return
    writeTextToClipboard(oauthState.userCode)
  }, [oauthState])

  const handleOAuthCopyUrl = useCallback(() => {
    if (!oauthState?.url) return
    writeTextToClipboard(oauthState.url)
  }, [oauthState])

  const handleLocalSetupSubmit = useCallback(async (config: { url: string; modelId: string; apiKey?: string }) => {
    setShowLocalSetup(false)
    await runtime.config.setLocalProviderConfig(config.url, config.modelId)
    if (config.apiKey) {
      await runtime.auth.setAuth('local', { type: 'api' as const, key: config.apiKey })
      trackProviderConnected({ providerId: 'local', authType: 'api' })
    } else {
      trackProviderConnected({ providerId: 'local', authType: 'none' })
    }
    const cfg = await runtime.config.loadConfig()
    cfg.providerOptions = cfg.providerOptions ?? {}
    cfg.providerOptions['local'] = { ...cfg.providerOptions['local'], baseUrl: config.url, modelId: config.modelId }
    await runtime.config.saveConfig(cfg)
    await reload()
    showEphemeral(`Connected to local model: ${config.modelId}`, theme.success)
    onAuthSuccess('local', 'Local')
  }, [showEphemeral, theme.success, onAuthSuccess, runtime, reload])

  const handleLocalSetupCancel = useCallback(() => {
    setShowLocalSetup(false)
    onAuthCancel()
  }, [onAuthCancel])

  const handleApiKeySubmit = useCallback(async (key: string) => {
    if (!apiKeySetup) return
    const auth = { type: 'api' as const, key }
    await runtime.auth.setAuth(apiKeySetup.provider.id, auth)
    trackProviderConnected({ providerId: apiKeySetup.provider.id, authType: 'api' })
    await reload()
    const providerName = apiKeySetup.provider.name
    const providerId = apiKeySetup.provider.id
    setApiKeySetup(null)
    showEphemeral(`Connected ${providerName} (API Key)`, theme.success)
    onAuthSuccess(providerId, providerName)
  }, [apiKeySetup, showEphemeral, theme.success, onAuthSuccess, runtime, reload])

  const handleApiKeyCancel = useCallback(() => {
    setApiKeySetup(null)
    onAuthCancel()
  }, [onAuthCancel])

  const cancelAll = useCallback(() => {
    oauthState?.cleanup?.()
    setOauthState(null)
    setApiKeySetup(null)
    setShowLocalSetup(false)
    setShowAuthMethodOverlay(false)
    setAuthMethodProvider(null)
  }, [oauthState])

  return {
    oauthState,
    apiKeySetup,
    showLocalSetup,
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
    handleLocalSetupSubmit,
    handleLocalSetupCancel,
    cancelAll,
  }
}