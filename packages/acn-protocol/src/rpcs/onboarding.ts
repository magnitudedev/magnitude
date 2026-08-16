import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { OnboardingError } from "../errors"
import { OnboardingState } from "../schemas/onboarding"
import { defineMirroredState } from "./mirrored-state"

export const OnboardingMirror = defineMirroredState("GetOnboardingState", {
  stateSchema: OnboardingState,
  errorSchema: OnboardingError,
})

export const CompleteOnboarding = Rpc.make("CompleteOnboarding", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: OnboardingError,
})
