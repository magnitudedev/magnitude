import { useCallback, useMemo } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  LocalModelsMirror,
  ModelSlotsMirror,
  OnboardingMirror,
  ProviderModelCatalogMirror,
  type OnboardingLocalModelSelection,
  type OnboardingFlowId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredState } from "./use-mirrored-state"

const ONBOARDING_LOCAL_MODEL_KEYS = [
  OnboardingMirror.id,
  LocalModelsMirror.id,
  ModelSlotsMirror.id,
  ProviderModelCatalogMirror.id,
] as const

export function useOnboardingState() {
  const client = useAgentClient()
  const completeAtom = useMemo(() => client.mutation("CompleteOnboardingFlow"), [client])
  const selectLocalModelAtom = useMemo(
    () => client.mutation("SelectOnboardingLocalModel"),
    [client],
  )
  const clearLocalModelAtom = useMemo(
    () => client.mutation("ClearOnboardingLocalModelSelection"),
    [client],
  )
  const state = Result.map(useMirroredState(OnboardingMirror), ({ state }) => state)
  const completeResult = useAtomValue(completeAtom)
  const selectLocalModelResult = useAtomValue(selectLocalModelAtom)
  const clearLocalModelResult = useAtomValue(clearLocalModelAtom)
  const completeMutation = useAtomSet(completeAtom)
  const selectLocalModelMutation = useAtomSet(selectLocalModelAtom)
  const clearLocalModelMutation = useAtomSet(clearLocalModelAtom)

  const complete = useCallback((flowId: OnboardingFlowId): void => {
    completeMutation({
      payload: { flowId },
      reactivityKeys: [OnboardingMirror.id],
    })
  }, [completeMutation])

  const selectLocalModel = useCallback((selection: OnboardingLocalModelSelection): void => {
    selectLocalModelMutation({
      payload: { selection },
      reactivityKeys: ONBOARDING_LOCAL_MODEL_KEYS,
    })
  }, [selectLocalModelMutation])

  const clearLocalModelSelection = useCallback((): void => {
    clearLocalModelMutation({
      payload: {},
      reactivityKeys: ONBOARDING_LOCAL_MODEL_KEYS,
    })
  }, [clearLocalModelMutation])

  return {
    state,
    completeResult,
    selectLocalModelResult,
    clearLocalModelResult,
    complete,
    selectLocalModel,
    clearLocalModelSelection,
  }
}
