import type { KeyEvent } from '@opentui/core'
import { useCallback, useEffect, useRef } from 'react'
import { createPasteIngestCoordinator, isPasteFallbackKey } from '../components/chat/paste/ingest-coordinator'
import type { PasteEventLike } from '../components/chat/paste/types'

interface UsePasteHandlerOptions {
  enabled?: boolean
  onPaste: (eventText?: string) => boolean | Promise<boolean>
}

export { isPasteFallbackKey }

export function createPasteFallbackController(
  onPaste: (eventText?: string) => boolean | Promise<boolean>,
  fallbackDelayMs = 25,
): {
  handlePasteKey: (key: KeyEvent, enabled?: boolean) => boolean
  handlePasteEvent: (event: PasteEventLike, enabled?: boolean) => void
  dispose: () => void
} {
  const coordinator = createPasteIngestCoordinator({
    fallbackDelayMs,
    requestFallbackPaste: () => onPaste(),
    onOutcome: (outcome) => {
      if (outcome.kind === 'native-event') {
        void onPaste(outcome.text)
      }
    },
  })

  return {
    handlePasteKey: (key: KeyEvent, enabled = true) => coordinator.handleKey(key, enabled),
    handlePasteEvent: (event: PasteEventLike, enabled = true) => coordinator.handleNativeEvent(event, enabled),
    dispose: () => coordinator.dispose(),
  }
}

export function usePasteHandler({ enabled = true, onPaste }: UsePasteHandlerOptions): {
  handlePasteKey: (key: KeyEvent) => boolean
  handlePasteEvent: (event: PasteEventLike) => void
} {
  const onPasteRef = useRef(onPaste)
  const controllerRef = useRef(
    createPasteFallbackController((eventText?: string) => onPasteRef.current(eventText)),
  )

  useEffect(() => {
    onPasteRef.current = onPaste
  }, [onPaste])

  useEffect(() => () => controllerRef.current.dispose(), [])

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
