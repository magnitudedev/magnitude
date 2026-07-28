import { useCallback, useRef, useState } from "react"
import { Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  type ModelOfferingTargetId,
  type ProviderModelId,
  type ReasoningEffort,
} from "@magnitudedev/sdk"
import { useLocalInferenceState } from "./use-local-inference-state"

export interface OnboardingModelChoice {
  readonly targetId: ModelOfferingTargetId
  readonly providerModelId: ProviderModelId
  readonly reasoningEffort: ReasoningEffort
}

export const useOnboardingModelSetup = (
  updateOnboarding: (completed: boolean) => Promise<unknown>,
) => {
  const local = useLocalInferenceState()
  const snapshot = Result.value(local.state)
  const view = Option.getOrNull(snapshot)
  const [submittedProviderModelId, setSubmittedProviderModelId] =
    useState<ProviderModelId | null>(null)
  const attemptRef = useRef(0)
  const {
    assignSlot,
    cancelModelDownload,
    clearSlot,
    downloadModel,
    loadModel,
  } = local

  const select = useCallback(async (choice: OnboardingModelChoice) => {
    const attempt = ++attemptRef.current
    setSubmittedProviderModelId(choice.providerModelId)
    try {
      await downloadModel(choice.targetId)
      if (attempt !== attemptRef.current) return
      await assignSlot(PRIMARY_SLOT_ID, {
        providerId: ProviderIdSchema.make("local"),
        providerModelId: choice.providerModelId,
        reasoningEffort: choice.reasoningEffort,
      })
      if (attempt !== attemptRef.current) return
      await loadModel(PRIMARY_SLOT_ID)
      if (attempt !== attemptRef.current) return
      await updateOnboarding(true)
    } catch {
      // Mutation Results and canonical mirrors own failure presentation.
    }
  }, [assignSlot, downloadModel, loadModel, updateOnboarding])

  const cancel = useCallback(() => {
    attemptRef.current += 1
    const submitted = submittedProviderModelId
    setSubmittedProviderModelId(null)
    if (submitted === null || view === null) return
    const candidate = view.models.recommendations._tag === "Ready"
      ? view.models.recommendations.catalog.find(({ providerModelId }) =>
          providerModelId === submitted)
      : undefined
    if (candidate?.download._tag === "Downloading") {
      cancelModelDownload(candidate.targetId)
      return
    }
    const primary = view.slots.slots.primary
    if (primary._tag === "ConfiguredLocal"
      && primary.selection.providerModelId === submitted) {
      clearSlot(PRIMARY_SLOT_ID)
    }
  }, [cancelModelDownload, clearSlot, submittedProviderModelId, view])

  return {
    local,
    view,
    submittedProviderModelId,
    select,
    cancel,
  }
}
