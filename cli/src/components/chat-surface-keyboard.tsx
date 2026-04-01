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
  pendingApproval: boolean
  onApprove: () => void
  onReject: () => void
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
  pendingApproval,
  onApprove,
  onReject,
}: ChatSurfaceKeyboardProps) {
  useKeyboard(
    useCallback((key: KeyEvent) => {
      if (key.defaultPrevented) return

      const isEscape = key.name === 'escape'
      const isCtrlC = key.ctrl && key.name === 'c' && !key.meta && !key.option

      if (isBlockingOverlayActive) return

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

      if (pendingApproval) {
        const isApprove = key.name === 'a' || key.name === 'enter' || key.name === 'return'
        const isReject = key.name === 'd' || isEscape
        if (isApprove && !key.ctrl && !key.meta && !key.option) {
          key.preventDefault()
          onApprove()
          return
        }
        if (isReject && !key.ctrl && !key.meta && !key.option) {
          key.preventDefault()
          onReject()
          return
        }
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
        killAllTimeoutRef.current = setTimeout(() => setNextEscWillKillAll(false), 2000)
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
      pendingApproval,
      onApprove,
      onReject,
    ]),
  )

  return null
}