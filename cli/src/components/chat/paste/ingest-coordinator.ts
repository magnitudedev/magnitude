import type { KeyEvent } from '@opentui/core'
import type {
  PasteEventLike,
  PasteIngestDropReason,
  PasteIngestOutcome,
  PasteIngestRequest,
  PendingPasteAttempt,
} from './types'

export function isPasteFallbackKey(key: KeyEvent): boolean {
  const keyName = (key.name ?? '').toLowerCase()
  const isCtrlV = key.ctrl && !key.meta
  const isCmdV = key.meta && !key.ctrl
  return !key.option && keyName === 'v' && (isCtrlV || isCmdV)
}

export function createPasteIngestCoordinator(args: {
  fallbackDelayMs?: number
  onOutcome: (outcome: PasteIngestOutcome) => void
  requestFallbackPaste: () => boolean | Promise<boolean>
}): {
  handleKey: (key: KeyEvent, enabled?: boolean) => boolean
  handleNativeEvent: (event: PasteEventLike, enabled?: boolean) => void
  dispose: () => void
} {
  const fallbackDelayMs = args.fallbackDelayMs ?? 25

  let fallbackTimer: ReturnType<typeof setTimeout> | null = null
  let pendingAttemptCleanupTimer: ReturnType<typeof setTimeout> | null = null
  let pendingAttempt: PendingPasteAttempt | null = null
  let deferredNativeEvent: PasteEventLike | null = null
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

  const emitDrop = (reason: PasteIngestDropReason) => args.onOutcome({ kind: 'dropped', reason })

  const handleShortcutFallbackRequest = () => {
    args.onOutcome({ kind: 'fallback-requested' })
  }

  const handleRequest = (request: PasteIngestRequest, enabled = true): boolean => {
    if (!enabled) {
      emitDrop('disabled')
      return false
    }

    if (request.type === 'shortcut') {
      const fakeKey = null
      void fakeKey
      clearFallbackTimer()
      clearPendingAttemptCleanupTimer()

      const attemptId = nextAttemptId++
      pendingAttempt = { id: attemptId, status: 'pending' }
      deferredNativeEvent = null

      fallbackTimer = setTimeout(() => {
        fallbackTimer = null
        if (!pendingAttempt || pendingAttempt.id !== attemptId || pendingAttempt.status !== 'pending') {
          return
        }

        pendingAttempt.status = 'fallback-in-flight'
        handleShortcutFallbackRequest()

        void Promise.resolve(args.requestFallbackPaste()).then((didInsert) => {
          if (!pendingAttempt || pendingAttempt.id !== attemptId || pendingAttempt.status !== 'fallback-in-flight') {
            return
          }

          if (didInsert) {
            pendingAttempt.status = 'fallback-succeeded'
          } else if (deferredNativeEvent) {
            const nativeEvent = deferredNativeEvent
            deferredNativeEvent = null
            clearPendingAttempt()
            args.onOutcome({
              kind: 'native-event',
              text: nativeEvent.text,
              replayedFromDeferred: true,
            })
            return
          } else {
            pendingAttempt.status = 'fallback-empty'
          }

          pendingAttemptCleanupTimer = setTimeout(() => {
            if (
              pendingAttempt?.id === attemptId &&
              (pendingAttempt.status === 'fallback-succeeded' || pendingAttempt.status === 'fallback-empty')
            ) {
              pendingAttempt = null
            }
            pendingAttemptCleanupTimer = null
          }, 50)
        })
      }, fallbackDelayMs)

      return true
    }

    if (pendingAttempt?.status === 'pending') {
      clearFallbackTimer()
      clearPendingAttempt()
      args.onOutcome({ kind: 'native-event', text: request.text, replayedFromDeferred: false })
      return true
    }

    if (pendingAttempt?.status === 'fallback-in-flight') {
      deferredNativeEvent = { text: request.text }
      return true
    }

    if (pendingAttempt?.status === 'fallback-succeeded') {
      clearPendingAttempt()
      emitDrop('native-duplicate-after-fallback-success')
      return true
    }

    if (pendingAttempt?.status === 'fallback-empty') {
      clearPendingAttempt()
      args.onOutcome({ kind: 'native-event', text: request.text, replayedFromDeferred: false })
      return true
    }

    args.onOutcome({ kind: 'native-event', text: request.text, replayedFromDeferred: false })
    return true
  }

  const handleKey = (key: KeyEvent, enabled = true): boolean => {
    if (!enabled) {
      emitDrop('disabled')
      return false
    }
    if (!isPasteFallbackKey(key)) {
      emitDrop('not-paste-shortcut')
      return false
    }

    key.preventDefault?.()
    return handleRequest({ type: 'shortcut' }, true)
  }

  const handleNativeEvent = (event: PasteEventLike, enabled = true): void => {
    void handleRequest({ type: 'native', text: event.text }, enabled)
  }

  const dispose = () => {
    clearFallbackTimer()
    clearPendingAttempt()
    deferredNativeEvent = null
  }

  return { handleKey, handleNativeEvent, dispose }
}
