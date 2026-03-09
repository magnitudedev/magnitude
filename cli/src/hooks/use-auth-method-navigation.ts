import { useState, useCallback } from 'react'
import type { KeyEvent } from '@opentui/core'
import type { AuthMethodDef } from '@magnitudedev/agent'

interface AuthMethodNavigationState {
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  handleKeyEvent: (key: KeyEvent) => boolean
}

/**
 * Manages keyboard navigation for the auth method selection overlay.
 * Handles Up/Down/Enter/Escape keys.
 */
export function useAuthMethodNavigation(
  methods: AuthMethodDef[],
  onSelect: (methodIndex: number) => void,
  onBack: () => void,
  isActive: boolean,
): AuthMethodNavigationState {
  const [selectedIndex, setSelectedIndex] = useState(0)

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (!isActive || methods.length === 0) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option
    const isEscape = key.name === 'escape'

    if (isEscape) {
      onBack()
      return true
    }

    if (isUp) {
      setSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }

    if (isDown) {
      setSelectedIndex(prev => Math.min(methods.length - 1, prev + 1))
      return true
    }

    if (isEnter) {
      onSelect(selectedIndex)
      return true
    }

    return false
  }, [isActive, methods, selectedIndex, onSelect, onBack])

  return {
    selectedIndex,
    setSelectedIndex,
    handleKeyEvent,
  }
}
