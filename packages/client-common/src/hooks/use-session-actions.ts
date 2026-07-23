import { useCallback } from "react"
import { useAtomSet } from "@effect-atom/atom-react"
import {
  selectedCwdAtom,
  pendingUserSubmitAtom,
} from "../state/session-atoms"
import { clearSystemMessages } from "../stores/system-message-store"
import { useDisplayViewControllerCore } from "../display-view-controller/hooks"
import { useDisplaySpeculator } from "../sync/index"

export interface StartNewSessionOptions {
  readonly cwd?: string | null
}

export interface SessionActions {
  readonly startNewSession: (options?: StartNewSessionOptions) => void
  readonly resumeSession: (sessionId: string) => void
}

/**
 * Shared session selection actions.
 *
 * Surfaces own presentation; this hook owns the client-local state transition
 * around selecting an existing session or returning to the new-session state.
 */
export function useSessionActions(): SessionActions {
  const controller = useDisplayViewControllerCore()
  const displaySpeculator = useDisplaySpeculator()
  const setSelectedCwd = useAtomSet(selectedCwdAtom)
  const setPendingUserSubmit = useAtomSet(pendingUserSubmitAtom)

  const resetSessionLocalState = useCallback(() => {
    setPendingUserSubmit(false)
    displaySpeculator.clear()
    clearSystemMessages()
  }, [
    displaySpeculator,
    setPendingUserSubmit,
  ])

  const startNewSession = useCallback((options?: StartNewSessionOptions) => {
    resetSessionLocalState()
    if (options && "cwd" in options) {
      setSelectedCwd(options.cwd ?? null)
    }
    controller.clearSession()
  }, [controller, resetSessionLocalState, setSelectedCwd])

  const resumeSession = useCallback((sessionId: string) => {
    resetSessionLocalState()
    controller.selectSession(sessionId)
  }, [controller, resetSessionLocalState])

  return { startNewSession, resumeSession }
}
