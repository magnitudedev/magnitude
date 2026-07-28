import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { LocalInferenceError, OnboardingError } from "../errors"
import { ProviderModelIdSchema } from "@magnitudedev/ai/provider/model"
import { CatalogCandidateIdSchema } from "../schemas/model-state"
import { OnboardingFlowId, OnboardingState } from "../schemas/onboarding"
import { defineMirroredState } from "./mirrored-state"

export const OnboardingLocalModelError = Schema.Union(LocalInferenceError, OnboardingError)

export const OnboardingLocalModelSelection = Schema.Union(
  Schema.TaggedStruct("CatalogCandidate", {
    candidateId: CatalogCandidateIdSchema,
  }),
  Schema.TaggedStruct("InstalledModel", {
    providerModelId: ProviderModelIdSchema,
  }),
)
export type OnboardingLocalModelSelection = typeof OnboardingLocalModelSelection.Type

export const OnboardingMirror = defineMirroredState("GetOnboardingState", {
  stateSchema: OnboardingState,
  errorSchema: OnboardingError,
})

export const CompleteOnboardingFlow = Rpc.make("CompleteOnboardingFlow", {
  payload: Schema.Struct({ flowId: OnboardingFlowId }),
  success: Schema.Struct({}),
  error: OnboardingError,
})

export const SelectOnboardingLocalModel = Rpc.make("SelectOnboardingLocalModel", {
  payload: Schema.Struct({ selection: OnboardingLocalModelSelection }),
  success: Schema.Struct({ providerModelId: ProviderModelIdSchema }),
  error: OnboardingLocalModelError,
})

export const ClearOnboardingLocalModelSelection = Rpc.make(
  "ClearOnboardingLocalModelSelection",
  {
    payload: Schema.Struct({}),
    success: Schema.Struct({}),
    error: OnboardingLocalModelError,
  },
)
