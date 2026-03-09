import { useState, useCallback } from 'react'
import type { RecentChat } from '../data/recent-chats'
import type { KeyEvent } from '@opentui/core'

interface RecentChatsNavigationState {
  selectedIndex: number
  setSelectedIndex: (index: number) => void
  /** Call from the parent's useKeyboard handler. Returns true if the key was consumed. */
  handleKeyEvent: (key: KeyEvent) => boolean
}

/**
 * Manages arrow-key + Enter navigation for a list of recent chats.
 *
 * Does NOT register its own useKeyboard — the caller integrates handleKeyEvent
 * into the existing keyboard priority chain (in app.tsx).
 */
export function useRecentChatsNavigation(
  chats: RecentChat[],
  onSelect: (chat: RecentChat) => void,
  isActive: boolean,
): RecentChatsNavigationState {
  const [selectedIndex, setSelectedIndex] = useState(0)

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (!isActive || chats.length === 0) return false

    const isUp = key.name === 'up' && !key.ctrl && !key.meta && !key.option
    const isDown = key.name === 'down' && !key.ctrl && !key.meta && !key.option
    const isEnter = (key.name === 'return' || key.name === 'enter') &&
      !key.shift && !key.ctrl && !key.meta && !key.option

    if (isUp) {
      setSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }

    if (isDown) {
      setSelectedIndex(prev => Math.min(chats.length - 1, prev + 1))
      return true
    }

    if (isEnter) {
      const chat = chats[selectedIndex]
      if (chat) {
        onSelect(chat)
      }
      return true
    }

    return false
  }, [isActive, chats, selectedIndex, onSelect])

  return {
    selectedIndex,
    setSelectedIndex,
    handleKeyEvent,
  }
}
