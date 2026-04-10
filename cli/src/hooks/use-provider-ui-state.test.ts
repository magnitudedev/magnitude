import { describe, expect, test, mock } from 'bun:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

const calls: string[] = []

mock.module('../providers/provider-runtime', () => ({
  useProviderRuntime: () => ({
    auth: {
      detectProviders: async () => {
        calls.push('runtime.detectProviders')
        return []
      },
    },
    state: {
      peek: async () => {
        calls.push('runtime.peek')
        return null
      },
    },
  }),
}))

mock.module('../providers/storage-provider', () => ({
  useStorage: () => ({
    config: {
      getSetupComplete: async () => {
        calls.push('storage.getSetupComplete')
        return true
      },
      getTelemetryEnabled: async () => {
        calls.push('storage.getTelemetryEnabled')
        return true
      },
      getLocalProviderConfig: async () => {
        calls.push('storage.getLocalProviderConfig')
        throw new Error('legacy local config path should not be called')
      },
    },
  }),
}))

const { useProviderUiState } = await import('./use-provider-ui-state')

describe('useProviderUiState', () => {
  test('reload uses provider/runtime + setup/telemetry paths and does not call legacy local config', async () => {
    let captured: ReturnType<typeof useProviderUiState> | null = null

    function Capture() {
      captured = useProviderUiState()
      return null
    }

    renderToStaticMarkup(React.createElement(Capture))
    expect(captured).not.toBeNull()

    await captured!.reload()

    expect(calls.includes('storage.getSetupComplete')).toBe(true)
    expect(calls.includes('storage.getTelemetryEnabled')).toBe(true)
    expect(calls.includes('runtime.detectProviders')).toBe(true)
    expect(calls.includes('runtime.peek')).toBe(true)
    expect(calls.includes('storage.getLocalProviderConfig')).toBe(false)
  })
})
