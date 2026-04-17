import { useState, useCallback } from 'react'
import type { KeyEvent } from '@opentui/core'
import type { MagnitudeSlot } from '@magnitudedev/agent'

interface SetupWizardNavState {
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  handleKeyEvent: (key: KeyEvent) => boolean
}

const SLOT_ORDER: MagnitudeSlot[] = ['lead', 'worker']

/**
 * Manages keyboard navigation for the setup wizard model confirmation step.
 * Items: 0 = "Start chatting/Continue", 1-2 = individual slots (lead, worker)
 */
export function useSetupWizardNavigation(
  onConfirm: () => void,
  onChangeSlot: (slot: MagnitudeSlot) => void,
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
      setSelectedIndex(prev => Math.min(2, prev + 1))
      return true
    }

    if (isEnter) {
      if (selectedIndex === 0) onConfirm()
      else if (selectedIndex >= 1 && selectedIndex <= 2) onChangeSlot(SLOT_ORDER[selectedIndex - 1])
      return true
    }

    return false
  }, [isActive, selectedIndex, onConfirm, onChangeSlot])

  return {
    selectedIndex,
    setSelectedIndex,
    handleKeyEvent,
  }
}
