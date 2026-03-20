import { useCallback, type RefObject } from 'react'
import type { KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'

interface ChatSurfaceKeyboardProps {
  status: 'idle' | 'streaming'
  hasRunningForks: boolean
  nextEscWillKillAll: boolean
  setNextEscWillKillAll: (next: boolean) => void
  killAllTimeoutRef: RefObject<NodeJS.Timeout | null>
  onInterrupt: () => void
  onInterruptAll: () => void
  inputText: string
  nextEscWillClearInput: boolean
  setNextEscWillClearInput: (next: boolean) => void
  clearInputTimeoutRef: RefObject<NodeJS.Timeout | null>
  onClearInput: () => void
  selectedFileOpen: boolean
  onCloseFilePanel: () => void
  bashMode: boolean
  onExitBashMode: () => void
  pendingApproval: boolean
  onApprove: () => void
  onReject: () => void
}

export function ChatSurfaceKeyboard({
  status,
  hasRunningForks,
  nextEscWillKillAll,
  setNextEscWillKillAll,
  killAllTimeoutRef,
  onInterrupt,
  onInterruptAll,
  inputText,
  nextEscWillClearInput,
  setNextEscWillClearInput,
  clearInputTimeoutRef,
  onClearInput,
  selectedFileOpen,
  onCloseFilePanel,
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

      if (isCtrlC && inputText.trim().length > 0) {
        key.preventDefault()
        onClearInput()
        return
      }

      if (isEscape && selectedFileOpen) {
        key.preventDefault()
        onCloseFilePanel()
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

      if (isEscape && inputText.length > 0) {
        key.preventDefault()
        if (nextEscWillClearInput) {
          onClearInput()
          setNextEscWillClearInput(false)
          if (clearInputTimeoutRef.current) {
            clearTimeout(clearInputTimeoutRef.current)
          }
        } else {
          setNextEscWillClearInput(true)
          if (clearInputTimeoutRef.current) {
            clearTimeout(clearInputTimeoutRef.current)
          }
          clearInputTimeoutRef.current = setTimeout(() => {
            setNextEscWillClearInput(false)
          }, 2000)
        }
      }
    }, [
      status,
      hasRunningForks,
      nextEscWillKillAll,
      setNextEscWillKillAll,
      killAllTimeoutRef,
      onInterrupt,
      onInterruptAll,
      inputText,
      nextEscWillClearInput,
      setNextEscWillClearInput,
      clearInputTimeoutRef,
      onClearInput,
      selectedFileOpen,
      onCloseFilePanel,
      bashMode,
      onExitBashMode,
      pendingApproval,
      onApprove,
      onReject,
    ]),
  )

  return null
}