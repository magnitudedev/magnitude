import type { KeyEvent } from '@opentui/core'
import { useCallback, useEffect, useRef } from 'react'

interface UsePasteHandlerOptions {
  enabled?: boolean
  onPaste: (eventText?: string) => void
}

interface PasteEventLike {
  text?: string
}

interface PendingPasteAttempt {
  id: number
  status: 'pending' | 'fallback-fired'
}

export function isPasteFallbackKey(key: KeyEvent): boolean {
  const keyName = (key.name ?? '').toLowerCase()
  const isCtrlV = key.ctrl && !key.meta
  const isCmdV = key.meta && !key.ctrl
  return !key.option && keyName === 'v' && (isCtrlV || isCmdV)
}

export function createPasteFallbackController(
  onPaste: (eventText?: string) => void,
  fallbackDelayMs = 25,
): {
  handlePasteKey: (key: KeyEvent, enabled?: boolean) => boolean
  handlePasteEvent: (event: PasteEventLike, enabled?: boolean) => void
  dispose: () => void
} {
  let fallbackTimer: ReturnType<typeof setTimeout> | null = null
  let pendingAttemptCleanupTimer: ReturnType<typeof setTimeout> | null = null
  let pendingAttempt: PendingPasteAttempt | null = null
  let nextAttemptId = 1

  const clearFallbackTimer = () => {
    if (!fallbackTimer) return
    clearTimeout(fallbackTimer)
    fallbackTimer = null
  }

  const clearPendingAttemptCleanupTimer = () => {
    if (!pendingAttemptCleanupTimer) return
    clearTimeout(pendingAttemptCleanupTimer)
    pendingAttemptCleanupTimer = null
  }

  const clearPendingAttempt = () => {
    pendingAttempt = null
    clearPendingAttemptCleanupTimer()
  }

  const handlePasteKey = (key: KeyEvent, enabled = true): boolean => {
    if (!enabled) return false
    if (!isPasteFallbackKey(key)) return false

    key.preventDefault?.()
    clearFallbackTimer()
    clearPendingAttemptCleanupTimer()

    const attemptId = nextAttemptId++
    pendingAttempt = { id: attemptId, status: 'pending' }

    fallbackTimer = setTimeout(() => {
      fallbackTimer = null
      if (!pendingAttempt || pendingAttempt.id !== attemptId || pendingAttempt.status !== 'pending') {
        return
      }

      pendingAttempt.status = 'fallback-fired'
      onPaste()

      pendingAttemptCleanupTimer = setTimeout(() => {
        if (pendingAttempt?.id === attemptId && pendingAttempt.status === 'fallback-fired') {
          pendingAttempt = null
        }
        pendingAttemptCleanupTimer = null
      }, 50)
    }, fallbackDelayMs)

    return true
  }

  const handlePasteEvent = (event: PasteEventLike, enabled = true) => {
    if (!enabled) return

    if (pendingAttempt?.status === 'pending') {
      clearFallbackTimer()
      clearPendingAttempt()
      onPaste(event.text)
      return
    }

    if (pendingAttempt?.status === 'fallback-fired') {
      clearPendingAttempt()
      return
    }

    onPaste(event.text)
  }

  const dispose = () => {
    clearFallbackTimer()
    clearPendingAttempt()
  }

  return { handlePasteKey, handlePasteEvent, dispose }
}

export function usePasteHandler({ enabled = true, onPaste }: UsePasteHandlerOptions): {
  handlePasteKey: (key: KeyEvent) => boolean
  handlePasteEvent: (event: PasteEventLike) => void
} {
  const controllerRef = useRef(createPasteFallbackController(onPaste))

  useEffect(() => {
    controllerRef.current = createPasteFallbackController(onPaste)
    return () => controllerRef.current.dispose()
  }, [onPaste])

  const handlePasteKey = useCallback(
    (key: KeyEvent): boolean => controllerRef.current.handlePasteKey(key, enabled),
    [enabled],
  )

  const handlePasteEvent = useCallback(
    (event: PasteEventLike) => controllerRef.current.handlePasteEvent(event, enabled),
    [enabled],
  )

  return { handlePasteKey, handlePasteEvent }
}
