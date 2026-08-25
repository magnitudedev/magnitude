import { Schema } from "effect"
import { Group, Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { OnboardingError } from "../errors"
import { OnboardingState } from "../schemas/onboarding"

/**
 * Authoritative onboarding state. Fresh until the ACN publishes a change for
 * it on `StreamChanges`; retained for the connection lifetime.
 */
const GetOnboardingState = Query.make("GetOnboardingState", {
  payload: Schema.Struct({}),
  success: OnboardingState,
  error: OnboardingError,
  staleTime: Infinity,
  gcTime: Infinity,
})

const CompleteOnboarding = Mutation.make("CompleteOnboarding", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: OnboardingError,
  synchronize: () => QueryClient.invalidate(GetOnboardingState.match()),
})

export const Onboarding = Group.make({ GetOnboardingState, CompleteOnboarding })
