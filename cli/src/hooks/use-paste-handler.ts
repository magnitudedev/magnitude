import type { KeyEvent } from '@opentui/core'
import { useCallback, useEffect, useRef } from 'react'

interface UsePasteHandlerOptions {
  enabled?: boolean
  onPaste: (eventText?: string) => void
}

interface PasteEventLike {
  text?: string
}

export function isPasteFallbackKey(key: KeyEvent): boolean {
  const keyName = (key.name ?? '').toLowerCase()
  return key.ctrl && !key.meta && !key.option && keyName === 'v'
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

  const clearFallbackTimer = () => {
    if (!fallbackTimer) return
    clearTimeout(fallbackTimer)
    fallbackTimer = null
  }

  const handlePasteKey = (key: KeyEvent, enabled = true): boolean => {
    if (!enabled) return false
    if (!isPasteFallbackKey(key)) return false
    key.preventDefault?.()
    clearFallbackTimer()
    fallbackTimer = setTimeout(() => {
      fallbackTimer = null
      onPaste()
    }, fallbackDelayMs)
    return true
  }

  const handlePasteEvent = (event: PasteEventLike, enabled = true) => {
    if (!enabled) return
    clearFallbackTimer()
    onPaste(event.text)
  }

  const dispose = () => {
    clearFallbackTimer()
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