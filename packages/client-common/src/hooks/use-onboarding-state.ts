import { useCallback, useMemo } from "react"
import { useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  LocalModelsMirror,
  ModelSlotsMirror,
  ProviderModelCatalogMirror,
  type CatalogCandidateId,
  type OnboardingFlowId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"

const ONBOARDING_LOCAL_MODEL_KEYS = [
  "onboarding",
  LocalModelsMirror.id,
  ModelSlotsMirror.id,
  ProviderModelCatalogMirror.id,
] as const

export function useOnboardingState() {
  const client = useAgentClient()
  const stateAtom = useMemo(
    () => client.query("GetOnboardingState", {}, { reactivityKeys: ["onboarding"] }),
    [client],
  )
  const completeAtom = useMemo(() => client.mutation("CompleteOnboardingFlow"), [client])
  const selectLocalModelAtom = useMemo(
    () => client.mutation("SelectOnboardingLocalModel"),
    [client],
  )
  const cancelLocalModelAtom = useMemo(
    () => client.mutation("CancelOnboardingLocalModelDownload"),
    [client],
  )
  const state = useAtomValue(stateAtom)
  const completeResult = useAtomValue(completeAtom)
  const selectLocalModelResult = useAtomValue(selectLocalModelAtom)
  const cancelLocalModelResult = useAtomValue(cancelLocalModelAtom)
  const completeMutation = useAtomSet(completeAtom)
  const selectLocalModelMutation = useAtomSet(selectLocalModelAtom)
  const cancelLocalModelMutation = useAtomSet(cancelLocalModelAtom)

  const complete = useCallback((flowId: OnboardingFlowId): void => {
    completeMutation({
      payload: { flowId },
      reactivityKeys: ["onboarding"],
    })
  }, [completeMutation])

  const selectLocalModel = useCallback((candidateId: CatalogCandidateId): void => {
    selectLocalModelMutation({
      payload: { candidateId },
      reactivityKeys: ONBOARDING_LOCAL_MODEL_KEYS,
    })
  }, [selectLocalModelMutation])

  const cancelLocalModelDownload = useCallback((): void => {
    cancelLocalModelMutation({
      payload: {},
      reactivityKeys: ONBOARDING_LOCAL_MODEL_KEYS,
    })
  }, [cancelLocalModelMutation])

  return {
    state,
    completeResult,
    selectLocalModelResult,
    cancelLocalModelResult,
    complete,
    selectLocalModel,
    cancelLocalModelDownload,
  }
}
