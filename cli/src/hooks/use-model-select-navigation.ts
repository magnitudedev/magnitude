import { useState, useCallback, useEffect } from 'react'
import type { KeyEvent } from '@opentui/core'

export interface ModelSelectItem {
  type: 'model'
  providerId: string
  providerName: string
  modelId: string
  modelName: string
  connected?: boolean
  selectable?: boolean
  recommended?: boolean
}

interface ModelSelectNavigationState {
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  items: ModelSelectItem[]
  handleKeyEvent: (key: KeyEvent) => boolean
}

function findNextSelectableIndex(items: ModelSelectItem[], startIndex: number, direction: -1 | 1): number {
  if (items.length === 0) return 0
  let index = Math.max(0, Math.min(items.length - 1, startIndex))

  while (index >= 0 && index < items.length) {
    if (items[index]?.selectable !== false) return index
    index += direction
  }

  return Math.max(0, Math.min(items.length - 1, startIndex))
}

export function useModelSelectNavigation(
  items: ModelSelectItem[],
  onSelect: (providerId: string, modelId: string) => void,
  isActive: boolean,
): ModelSelectNavigationState {
  const [selectedIndex, setSelectedIndex] = useState(0)

  useEffect(() => {
    if (items.length === 0) {
      setSelectedIndex(0)
      return
    }

    setSelectedIndex((prev) => {
      const clamped = Math.max(0, Math.min(items.length - 1, prev))
      if (items[clamped]?.selectable !== false) return clamped
      return findNextSelectableIndex(items, clamped, 1)
    })
  }, [items])

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (!isActive || items.length === 0) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option

    if (isUp) {
      setSelectedIndex(prev => findNextSelectableIndex(items, prev - 1, -1))
      return true
    }

    if (isDown) {
      setSelectedIndex(prev => findNextSelectableIndex(items, prev + 1, 1))
      return true
    }

    if (isEnter) {
      const item = items[selectedIndex]
      if (item && item.selectable !== false) {
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