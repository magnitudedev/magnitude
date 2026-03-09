import { useState, useCallback } from 'react'
import type { KeyEvent } from '@opentui/core'

interface SetupWizardNavState {
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  handleKeyEvent: (key: KeyEvent) => boolean
}

/**
 * Manages keyboard navigation for the setup wizard model confirmation step.
 * Items: 0 = "Start chatting/Continue", 1 = "Change primary", 2 = "Change secondary", 3 = "Change browser"
 */
export function useSetupWizardNavigation(
  onConfirm: () => void,
  onChangePrimary: () => void,
  onChangeSecondary: () => void,
  onChangeBrowser: () => void,
  isActive: boolean,
): SetupWizardNavState {
  const [selectedIndex, setSelectedIndex] = useState(0)

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (!isActive) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option

    if (isUp) {
      setSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }

    if (isDown) {
      setSelectedIndex(prev => Math.min(3, prev + 1))
      return true
    }

    if (isEnter) {
      if (selectedIndex === 0) onConfirm()
      else if (selectedIndex === 1) onChangePrimary()
      else if (selectedIndex === 2) onChangeSecondary()
      else if (selectedIndex === 3) onChangeBrowser()
      return true
    }

    return false
  }, [isActive, selectedIndex, onConfirm, onChangePrimary, onChangeSecondary, onChangeBrowser])

  return {
    selectedIndex,
    setSelectedIndex,
    handleKeyEvent,
  }
}
