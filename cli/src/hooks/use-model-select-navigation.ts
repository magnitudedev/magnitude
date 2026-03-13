import { useState, useCallback } from 'react'
import type { KeyEvent } from '@opentui/core'

export interface ModelSelectItem {
  type: 'model'
  providerId: string
  providerName: string
  modelId: string
  modelName: string
}

interface ModelSelectNavigationState {
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  items: ModelSelectItem[]
  handleKeyEvent: (key: KeyEvent) => boolean
}

export function useModelSelectNavigation(
  items: ModelSelectItem[],
  onSelect: (providerId: string, modelId: string) => void,
  isActive: boolean,
): ModelSelectNavigationState {
  const [selectedIndex, setSelectedIndex] = useState(0)

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (!isActive || items.length === 0) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option

    if (isUp) {
      setSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }

    if (isDown) {
      setSelectedIndex(prev => Math.min(items.length - 1, prev + 1))
      return true
    }

    if (isEnter) {
      const item = items[selectedIndex]
      if (item) {
        onSelect(item.providerId, item.modelId)
      }
      return true
    }

    return false
  }, [isActive, items, selectedIndex, onSelect])

  return {
    selectedIndex,
    setSelectedIndex,
    items,
    handleKeyEvent,
  }
}