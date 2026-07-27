import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { LocalInferenceError, OnboardingError } from "../errors"
import { CatalogCandidateIdSchema } from "../schemas/model-state"
import { OnboardingFlowId, OnboardingState } from "../schemas/onboarding"

export const OnboardingLocalModelError = Schema.Union(LocalInferenceError, OnboardingError)

export const GetOnboardingState = Rpc.make("GetOnboardingState", {
  payload: Schema.Struct({}),
  success: OnboardingState,
  error: OnboardingError,
})

export const CompleteOnboardingFlow = Rpc.make("CompleteOnboardingFlow", {
  payload: Schema.Struct({ flowId: OnboardingFlowId }),
  success: Schema.Struct({}),
  error: OnboardingError,
})

export const SelectOnboardingLocalModel = Rpc.make("SelectOnboardingLocalModel", {
  payload: Schema.Struct({ candidateId: CatalogCandidateIdSchema }),
  success: Schema.Struct({}),
  error: OnboardingLocalModelError,
})

export const CancelOnboardingLocalModelDownload = Rpc.make(
  "CancelOnboardingLocalModelDownload",
  {
    payload: Schema.Struct({}),
    success: Schema.Struct({}),
    error: OnboardingLocalModelError,
  },
)
