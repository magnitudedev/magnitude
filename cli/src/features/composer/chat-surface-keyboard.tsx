import { useCallback, type RefObject } from 'react'
import type { KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'

interface ChatSurfaceKeyboardProps {
  status: 'idle' | 'streaming'
  hasRunningForks: boolean
  isBlockingOverlayActive: boolean
  nextEscWillKillAll: boolean
  setNextEscWillKillAll: (next: boolean) => void
  killAllTimeoutRef: RefObject<NodeJS.Timeout | null>
  onInterrupt: () => void
  onInterruptAll: () => void
  composerHasContent: boolean
  onClearInput: () => void
  bashMode: boolean
  onExitBashMode: () => void
  onToggleAutopilot?: () => void
  thinkingOpen: boolean
  thinkingOptionCount: number
  onToggleThinking: () => void
  onMoveThinking: (direction: -1 | 1) => void
  onApplyThinking: () => void
  onCancelThinking: () => void
}

export function ChatSurfaceKeyboard({
  status,
  hasRunningForks,
  isBlockingOverlayActive,
  nextEscWillKillAll,
  setNextEscWillKillAll,
  killAllTimeoutRef,
  onInterrupt,
  onInterruptAll,
  composerHasContent,
  onClearInput,
  bashMode,
  onExitBashMode,
  onToggleAutopilot,
  thinkingOpen,
  thinkingOptionCount,
  onToggleThinking,
  onMoveThinking,
  onApplyThinking,
  onCancelThinking,
}: ChatSurfaceKeyboardProps) {
  useKeyboard(
    useCallback((key: KeyEvent) => {
      if (key.defaultPrevented) return

      const isEscape = key.name === 'escape'
      const isCtrlC = key.ctrl && key.name === 'c' && !key.meta && !key.option
      const isCtrlT = key.ctrl && key.name === 't' && !key.meta && !key.option

      if (isBlockingOverlayActive) return

      if (isCtrlT && thinkingOptionCount > 0) {
        key.preventDefault()
        onToggleThinking()
        return
      }

      if (thinkingOpen) {
        if (isEscape) {
          key.preventDefault()
          onCancelThinking()
          return
        }
        if (key.name === 'up' || key.name === 'k') {
          key.preventDefault()
          onMoveThinking(-1)
          return
        }
        if (key.name === 'down' || key.name === 'j') {
          key.preventDefault()
          onMoveThinking(1)
          return
        }
        if (key.name === 'return' || key.name === 'enter') {
          key.preventDefault()
          onApplyThinking()
          return
        }
        return
      }

      const isCtrlA = key.ctrl && key.name === 'a' && !key.meta && !key.option
      if (isCtrlA && onToggleAutopilot) {
        key.preventDefault()
        onToggleAutopilot()
        return
      }

      if (isCtrlC && composerHasContent) {
        key.preventDefault()
        onClearInput()
        return
      }

      if (isEscape && bashMode) {
        key.preventDefault()
        onExitBashMode()
        return
      }

      if (isEscape && nextEscWillKillAll) {
        key.preventDefault()
        onInterruptAll()
        setNextEscWillKillAll(false)
        if (killAllTimeoutRef.current) {
          clearTimeout(killAllTimeoutRef.current)
        }
        return
      }

      if (isEscape && hasRunningForks) {
        key.preventDefault()
        if (status === 'streaming') onInterrupt()
        setNextEscWillKillAll(true)
        if (killAllTimeoutRef.current) {
          clearTimeout(killAllTimeoutRef.current)
        }
        killAllTimeoutRef.current = setTimeout(() => setNextEscWillKillAll(false), 5000)
        return
      }

      if ((isEscape || isCtrlC) && status === 'streaming') {
        key.preventDefault()
        onInterrupt()
        return
      }


    }, [
      status,
      hasRunningForks,
      isBlockingOverlayActive,
      nextEscWillKillAll,
      setNextEscWillKillAll,
      killAllTimeoutRef,
      onInterrupt,
      onInterruptAll,
      composerHasContent,
      onClearInput,
      bashMode,
      onExitBashMode,
      onToggleAutopilot,
      thinkingOpen,
      thinkingOptionCount,
      onToggleThinking,
      onMoveThinking,
      onApplyThinking,
      onCancelThinking,
    ]),
  )

  return null
}
