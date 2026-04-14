import { useCallback } from 'react'
import type { KeyEvent } from '@opentui/core'

export type SettingsTab = 'provider' | 'model' | 'skillset'

const TAB_ORDER: SettingsTab[] = ['provider', 'model', 'skillset']

/**
 * Combines tab switching (left/right) with delegation to the active tab's
 * list navigation (up/down/enter).
 */
export function useSettingsNavigation(
  activeTab: SettingsTab,
  onTabChange: (tab: SettingsTab) => void,
  modelHandleKeyEvent: (key: KeyEvent) => boolean,
  providerHandleKeyEvent: (key: KeyEvent) => boolean,
  skillsetHandleKeyEvent: (key: KeyEvent) => boolean,
  isActive: boolean,
): { handleKeyEvent: (key: KeyEvent) => boolean } {

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (!isActive) return false

    const plain = !key.ctrl && !key.meta && !key.option

    // Tab switching
    if (key.name === 'left' && plain) {
      const idx = TAB_ORDER.indexOf(activeTab)
      if (idx > 0) {
        onTabChange(TAB_ORDER[idx - 1])
        return true
      }
    }
    if (key.name === 'right' && plain) {
      const idx = TAB_ORDER.indexOf(activeTab)
      if (idx < TAB_ORDER.length - 1) {
        onTabChange(TAB_ORDER[idx + 1])
        return true
      }
    }

    // Delegate to active tab's list navigation
    if (activeTab === 'model') {
      return modelHandleKeyEvent(key)
    } else if (activeTab === 'skillset') {
      return skillsetHandleKeyEvent(key)
    } else {
      return providerHandleKeyEvent(key)
    }
  }, [isActive, activeTab, onTabChange, modelHandleKeyEvent, providerHandleKeyEvent, skillsetHandleKeyEvent])

  return { handleKeyEvent }
}
