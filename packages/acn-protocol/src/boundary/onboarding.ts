import { Rpc } from "@effect/rpc"
import { replaySafe } from "../transport/recovery"
import { Schema } from "effect"
import { OnboardingError } from "../errors"
import { OnboardingState } from "../schemas/onboarding"

const GetOnboardingState = Rpc.make("GetOnboardingState", {
  payload: Schema.Struct({}),
  success: OnboardingState,
  error: OnboardingError,
}).pipe(replaySafe)

const CompleteOnboarding = Rpc.make("CompleteOnboarding", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: OnboardingError,
}).pipe(replaySafe)

export const Onboarding = {
  getOnboardingState: GetOnboardingState,
  completeOnboarding: CompleteOnboarding,
}
