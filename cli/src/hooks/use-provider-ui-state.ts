import { useCallback, useEffect, useRef, useState } from 'react'
import type { ModelSelection } from '@magnitudedev/providers'
import type { DetectedProvider, MagnitudeSlot } from '@magnitudedev/agent'
import { MAGNITUDE_SLOTS } from '@magnitudedev/agent'
import { useProviderRuntime } from '../providers/provider-runtime'
import { useStorage } from '../providers/storage-provider'

interface ProviderUiState {
  detectedProviders: DetectedProvider[]
  slotModels: Record<MagnitudeSlot, ModelSelection | null>
  setupComplete: boolean
  telemetryEnabled: boolean
}

export function useProviderUiState() {
  const runtime = useProviderRuntime()
  const storage = useStorage()
  const [state, setState] = useState<ProviderUiState | null>(null)
  const versionRef = useRef(0)

  const reload = useCallback(async () => {
    const version = ++versionRef.current
    const [setupComplete, telemetryEnabled, runtimeDetectedProviders, ...slotStates] = await Promise.all([
      storage.config.getSetupComplete(),
      storage.config.getTelemetryEnabled(),
      runtime.auth.detectProviders(),
      ...MAGNITUDE_SLOTS.map(slot => runtime.state.peek(slot)),
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

    const slotModels = Object.fromEntries(
      MAGNITUDE_SLOTS.map((slot, i) => {
        const s = slotStates[i]
        return [slot, s?.model ? { providerId: s.model.providerId, modelId: s.model.id } : null]
      })
    ) as Record<MagnitudeSlot, ModelSelection | null>

    setState({
      slotModels,
      detectedProviders,
      setupComplete,
      telemetryEnabled,
    })
  }, [runtime, storage])

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
