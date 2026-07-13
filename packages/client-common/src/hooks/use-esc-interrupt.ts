/**
 * Esc interrupt composable — shared between web, desktop, and CLI.
 *
 * Handles the interrupt + fork-pop logic that both apps share:
 * - If fork stack is non-empty: pop the top fork (close worker detail)
 * - If fork stack is empty:
 *   - First Esc: set nextEscWillKillAll hint
 *   - Second Esc (within 400ms): dispatch interrupt-all
 *
 * Does NOT handle bash-mode-exit or overlay-close — those are app-specific
 * keyboard concerns handled by each app's keyboard handler.
 *
 * The *trigger* is app-specific (OpenTUI useKeyboard vs DOM keydown).
 * Each app's keyboard handler calls `handleEscKey` when Escape is pressed.
 */
import { useAtomValue, useAtomSet } from "@effect-atom/atom-react"
import { nextEscWillKillAllAtom } from "../state/session-atoms"
import { useDisplayViewController } from "../display-view-controller/hooks"

export interface UseEscInterruptResult {
  /** Call this when Escape is pressed. Returns true if handled (fork pop or interrupt). */
  handleEscKey: () => boolean
  /** The current nextEscWillKillAll hint state */
  nextEscWillKillAll: boolean
}

/**
 * Shared Esc handling composable.
 * Takes `onInterruptAll` callback — the app provides the actual interrupt dispatch
 * (e.g. dispatching a custom event or calling a mutation).
 */
export function useEscInterrupt(onInterruptAll: () => void): UseEscInterruptResult {
  const { expandedForkStack, popFork } = useDisplayViewController()
  const nextEscWillKillAll = useAtomValue(nextEscWillKillAllAtom)
  const setNextEscWillKillAll = useAtomSet(nextEscWillKillAllAtom)

  let lastEscTime = 0

  function handleEscKey(): boolean {
    const now = Date.now()
    const isDoubleEsc = now - lastEscTime < 400
    lastEscTime = now

    // Pop fork stack if non-empty
    if (expandedForkStack.length > 0) {
      popFork()
      return true
    }

    // Double Esc → interrupt all
    if (isDoubleEsc) {
      setNextEscWillKillAll(false)
      onInterruptAll()
      return true
    }

    // First Esc → show hint
    setNextEscWillKillAll(true)
    setTimeout(() => {
      setNextEscWillKillAll(false)
    }, 400)
    return true
  }

  return { handleEscKey, nextEscWillKillAll }
}
