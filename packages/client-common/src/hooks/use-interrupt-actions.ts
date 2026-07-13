/**
 * Interrupt actions hook — shared between web, desktop, and CLI.
 *
 * Provides `interrupt` (single fork or root) and `interruptAll` (all workers)
 * mutation dispatchers. Both apps use this identically.
 */
import { useAtomSet } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { useSelectedSessionId } from "../display-view-controller/hooks"

export interface UseInterruptActionsResult {
  /** Interrupt a specific fork, or the root agent if forkId is null */
  interrupt: (forkId?: string | null) => void
  /** Interrupt all workers in the session */
  interruptAll: () => void
}

export function useInterruptActions(): UseInterruptActionsResult {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const interruptMutation = useAtomSet(client.mutation("Interrupt"))

  function interrupt(forkId?: string | null): void {
    if (!selectedSessionId) return
    interruptMutation({
      payload: {
        sessionId: selectedSessionId,
        target: { _tag: "fork", forkId: forkId ?? null },
      },
    })
  }

  function interruptAll(): void {
    if (!selectedSessionId) return
    interruptMutation({
      payload: {
        sessionId: selectedSessionId,
        target: { _tag: "all" },
      },
    })
  }

  return { interrupt, interruptAll }
}
