import { Data, Effect, Schema } from "effect"
import { Group, Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { OnboardingError } from "../errors"
import { MirroredSnapshotSchema } from "../schemas/mirrored-state"
import { OnboardingState } from "../schemas/onboarding"

/**
 * Authoritative onboarding state. Fresh until the ACN publishes a change for
 * it on `StreamChanges`; retained for the connection lifetime.
 */
const GetOnboardingState = Query.make("GetOnboardingState", {
  payload: Schema.Struct({}),
  success: MirroredSnapshotSchema(OnboardingState),
  error: OnboardingError,
  staleTime: Infinity,
  gcTime: Infinity,
})

/** Completion was acknowledged but the state did not report completed. */
export class OnboardingPersistenceSynchronizationFailed extends Data.TaggedError(
  "OnboardingPersistenceSynchronizationFailed",
)<{}> {}

const CompleteOnboarding = Mutation.make("CompleteOnboarding", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: OnboardingError,
  synchronize: () => QueryClient.invalidate(GetOnboardingState.match()).pipe(
    Effect.zipRight(QueryClient.fetch(GetOnboardingState, {})),
    Effect.filterOrFail(
      ({ state }) => state.completed,
      () => new OnboardingPersistenceSynchronizationFailed(),
    ),
    Effect.asVoid,
  ),
})

export const Onboarding = Group.make({ GetOnboardingState, CompleteOnboarding })
