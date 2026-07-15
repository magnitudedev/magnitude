import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { OnboardingError } from "../errors"
import { OnboardingFlowId, OnboardingState } from "../schemas/onboarding"

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
