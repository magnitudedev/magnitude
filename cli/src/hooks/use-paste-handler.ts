import type { KeyEvent } from '@opentui/core'
import { useCallback, useRef } from 'react'
import { createPasteIngestCoordinator, isPasteFallbackKey } from '@magnitudedev/client-common'
import type { PasteEventLike } from '@magnitudedev/client-common'

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

/**
 * Paste handler hook — no useEffect.
 *
 * Uses refs to hold mutable state (onPaste callback, controller).
 * The controller is created once and disposed via a finalizer ref
 * that runs on unmount through the renderer's cleanup cycle.
 * The onPaste ref is updated directly during render (no effect needed).
 */
export function usePasteHandler({ enabled = true, onPaste }: UsePasteHandlerOptions): {
  handlePasteKey: (key: KeyEvent) => boolean
  handlePasteEvent: (event: PasteEventLike) => void
} {
  const onPasteRef = useRef(onPaste)
  // Update ref directly during render — safe, no re-render trigger
  onPasteRef.current = onPaste

  const controllerRef = useRef<ReturnType<typeof createPasteFallbackController> | null>(null)
  if (controllerRef.current === null) {
    controllerRef.current = createPasteFallbackController((eventText?: string) => onPasteRef.current(eventText))
  }

  const handlePasteKey = useCallback(
    (key: KeyEvent): boolean => controllerRef.current!.handlePasteKey(key, enabled),
    [enabled],
  )

  const handlePasteEvent = useCallback(
    (event: PasteEventLike) => controllerRef.current!.handlePasteEvent(event, enabled),
    [enabled],
  )

  return { handlePasteKey, handlePasteEvent }
}
