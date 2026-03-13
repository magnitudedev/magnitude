import { useCallback, useEffect, useRef, useState } from 'react'
import type { MagnitudeConfig, ModelSelection } from '@magnitudedev/providers'
import type { DetectedProvider } from '@magnitudedev/agent'
import { useProviderRuntime } from '../providers/provider-runtime'

type LocalProviderConfig = Awaited<
  ReturnType<ReturnType<typeof useProviderRuntime>['config']['getLocalProviderConfig']>
>

interface ProviderUiState {
  config: MagnitudeConfig
  primaryModel: ModelSelection | null
  secondaryModel: ModelSelection | null
  browserModel: ModelSelection | null
  detectedProviders: DetectedProvider[]
  localProviderConfig: LocalProviderConfig
}

export function useProviderUiState() {
  const runtime = useProviderRuntime()
  const [state, setState] = useState<ProviderUiState | null>(null)
  const versionRef = useRef(0)

  const reload = useCallback(async () => {
    const version = ++versionRef.current
    const [config, primarySlot, secondarySlot, browserSlot, runtimeDetectedProviders, localProviderConfig] = await Promise.all([
      runtime.config.loadConfig(),
      runtime.state.peek('primary'),
      runtime.state.peek('secondary'),
      runtime.state.peek('browser'),
      runtime.auth.detectProviders(),
      runtime.config.getLocalProviderConfig(),
    ])

    const detectedProviders: DetectedProvider[] = runtimeDetectedProviders.map((entry) => {
      const preferredMethod =
        entry.authMethods.find((method) => method.source === 'stored' && method.connected) ??
        entry.authMethods.find((method) => method.source === 'env' && method.connected) ??
        entry.authMethods.find((method) => method.source === 'none' && method.connected)

      return {
        provider: entry.provider,
        auth: preferredMethod?.auth ?? null,
        source: preferredMethod?.source ?? 'none',
      }
    })

    if (version !== versionRef.current) return

    setState({
      config,
      primaryModel: primarySlot?.model
        ? { providerId: primarySlot.model.providerId, modelId: primarySlot.model.id }
        : null,
      secondaryModel: secondarySlot?.model
        ? { providerId: secondarySlot.model.providerId, modelId: secondarySlot.model.id }
        : null,
      browserModel: browserSlot?.model
        ? { providerId: browserSlot.model.providerId, modelId: browserSlot.model.id }
        : null,
      detectedProviders,
      localProviderConfig,
    })
  }, [runtime])

  useEffect(() => {
    reload().catch((error) => {
      console.error('Failed to load provider UI state', error)
    })
  }, [reload])

  return {
    state,
    reload,
  }
}