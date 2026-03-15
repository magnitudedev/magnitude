import type { KeyEvent } from '@opentui/core'
import { useCallback } from 'react'

interface UsePasteHandlerOptions {
  enabled?: boolean
  onPaste: (eventText?: string) => void
}

interface PasteEventLike {
  text?: string
}

export function usePasteHandler({ enabled = true, onPaste }: UsePasteHandlerOptions): {
  handlePasteKey: (key: KeyEvent) => boolean
  handlePasteEvent: (event: PasteEventLike) => void
} {
  const handlePasteKey = useCallback(
    (key: KeyEvent): boolean => {
      if (!enabled) return false

      const keyName = (key.name ?? '').toLowerCase()
      const isCtrlV = key.ctrl && !key.meta && !key.option && keyName === 'v'
      const isCmdV = key.meta && !key.ctrl && !key.option && keyName === 'v'

      if (!isCtrlV && !isCmdV) return false

      key.preventDefault?.()
      onPaste()
      return true
    },
    [enabled, onPaste],
  )

  const handlePasteEvent = useCallback(
    (event: PasteEventLike) => {
      if (!enabled) return
      onPaste(event.text)
    },
    [enabled, onPaste],
  )

  return { handlePasteKey, handlePasteEvent }
}