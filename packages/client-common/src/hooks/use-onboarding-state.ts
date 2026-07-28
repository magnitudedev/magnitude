import { useCallback, useMemo } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  OnboardingMirror,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredState } from "./use-mirrored-state"

export function useOnboardingState() {
  const client = useAgentClient()
  const updateAtom = useMemo(() => client.mutation("UpdateOnboardingState"), [client])
  const state = Result.map(useMirroredState(OnboardingMirror), ({ state }) => state)
  const updateResult = useAtomValue(updateAtom)
  const updateMutation = useAtomSet(updateAtom, { mode: "promise" })

  const update = useCallback((completed: boolean) =>
    updateMutation({
      payload: { completed },
      reactivityKeys: [OnboardingMirror.id],
    }), [updateMutation])

  return {
    state,
    updateResult,
    update,
  }
}
