import { useState, useCallback } from 'react'
import type { KeyEvent } from '@opentui/core'
import type { ProviderDefinition } from '@magnitudedev/agent'

interface ProviderSelectNavigationState {
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  handleKeyEvent: (key: KeyEvent) => boolean
}

/**
 * Manages keyboard navigation for the provider select overlay.
 * Handles Up/Down/Enter keys.
 */
export function useProviderSelectNavigation(
  providers: ProviderDefinition[],
  onSelect: (providerId: string) => void,
  isActive: boolean,
): ProviderSelectNavigationState {
  const [selectedIndex, setSelectedIndex] = useState(0)

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (!isActive || providers.length === 0) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option

    if (isUp) {
      setSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }

    if (isDown) {
      setSelectedIndex(prev => Math.min(providers.length - 1, prev + 1))
      return true
    }

    if (isEnter) {
      const provider = providers[selectedIndex]
      if (provider) {
        onSelect(provider.id)
      }
      return true
    }

    return false
  }, [isActive, providers, selectedIndex, onSelect])

  return {
    selectedIndex,
    setSelectedIndex,
    handleKeyEvent,
  }
}
