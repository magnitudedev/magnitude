import { useCallback, useMemo } from "react"
import { useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import type { OnboardingFlowId } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"

export function useOnboardingState() {
  const client = useAgentClient()
  const stateAtom = useMemo(
    () => client.query("GetOnboardingState", {}, { reactivityKeys: ["onboarding"] }),
    [client],
  )
  const completeAtom = useMemo(() => client.mutation("CompleteOnboardingFlow"), [client])
  const state = useAtomValue(stateAtom)
  const completeResult = useAtomValue(completeAtom)
  const completeMutation = useAtomSet(completeAtom)

  const complete = useCallback((flowId: OnboardingFlowId): void => {
    completeMutation({
      payload: { flowId },
      reactivityKeys: ["onboarding"],
    })
  }, [completeMutation])

  return { state, completeResult, complete }
}
