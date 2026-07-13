/**
 * Terminal-global keyboard handling (spec §2.11, §5.6 AppShell) — the
 * OpenTUI-side trigger layer. Chord → intent mapping is app-specific;
 * all effects go through atoms so features react independently:
 *
 * - Ctrl+C: exit (guarded — not while composing or streaming)
 * - Ctrl+R: toggle recent chats overlay
 * - Ctrl+T: toggle default/transcript display mode
 * - Error CTA chords: dispatch the most recent actionable inline error
 */
import { useCallback } from 'react'
import { Option } from 'effect'
import { useKeyboard } from '@opentui/react'
import type { KeyEvent } from '@opentui/core'
import { useAtomValue, useAtomSet } from '@effect-atom/atom-react'
import {
  useDisplayState,
  getFork,
  useDisplayViewController,
  settingsOpenAtom,
  usageOpenAtom,
  composerHasContentAtom,
} from '@magnitudedev/client-common'
import { showRecentChatsOverlayAtom } from '../state/cli-atoms'
import { matchKeyToChord } from '../utils/chord'
import type { ActionId } from '../types/ui-actions'

export interface TerminalKeyboardParams {
  dispatchErrorAction: (actionId: ActionId) => void
}

export function useTerminalKeyboard({ dispatchErrorAction }: TerminalKeyboardParams): void {
  const composerHasContent = useAtomValue(composerHasContentAtom)
  const { expandedForkStack, togglePresentationMode } = useDisplayViewController()
  const showRecentChats = useAtomValue(showRecentChatsOverlayAtom)
  const setShowRecentChats = useAtomSet(showRecentChatsOverlayAtom)
  const settingsOpen = useAtomValue(settingsOpenAtom)
  const usageOpen = useAtomValue(usageOpenAtom)

  const rootMode = useDisplayState((state) => getFork(state, null)?.mode ?? 'idle')
  const latestErrorCta = useDisplayState((state) => {
    const messages = getFork(state, null)?.messages
    if (!messages) return null
    for (let i = messages.order.length - 1; i >= 0; i--) {
      const m = messages.byId[messages.order[i]]
      if (m && m.type === 'error') {
        const cta = Option.getOrNull(m.cta)
        if (cta?.kind === 'action') return cta
      }
    }
    return null
  })

  const overlayActive = showRecentChats || settingsOpen || usageOpen || expandedForkStack.length > 0
  const canToggleRecentChats = !settingsOpen && !usageOpen && expandedForkStack.length === 0

  useKeyboard(
    useCallback((key: KeyEvent) => {
      if (key.defaultPrevented) return

      const isCtrlC = key.ctrl && key.name === 'c' && !key.meta && !key.option
      const isCtrlR = key.ctrl && key.name === 'r' && !key.meta && !key.option
      const isCtrlT = key.ctrl && key.name === 't' && !key.meta && !key.option

      if (isCtrlC) {
        if (composerHasContent) return
        if (rootMode === 'streaming') return
        key.preventDefault()
        process.kill(process.pid, 'SIGINT')
        return
      }

      if (isCtrlR) {
        if (!canToggleRecentChats) return
        key.preventDefault()
        setShowRecentChats((prev: boolean) => !prev)
        return
      }

      if (isCtrlT) {
        key.preventDefault()
        togglePresentationMode()
        return
      }

      if (!overlayActive && latestErrorCta && latestErrorCta.kind === 'action') {
        const chord = matchKeyToChord(key)
        if (chord === latestErrorCta.chord) {
          key.preventDefault()
          dispatchErrorAction(latestErrorCta.actionId as ActionId)
        }
      }
    }, [
      composerHasContent,
      rootMode,
      canToggleRecentChats,
      overlayActive,
      latestErrorCta,
      setShowRecentChats,
      togglePresentationMode,
      dispatchErrorAction,
    ]),
  )
}
