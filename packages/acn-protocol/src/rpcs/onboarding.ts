import { Data, Effect, Schema } from "effect"
import { QueryClient } from "@magnitudedev/effect-query"
import { Acn } from "../boundary"
import { OnboardingError } from "../errors"
import { MirroredSnapshotSchema } from "../schemas/mirrored-state"
import { OnboardingState } from "../schemas/onboarding"

/**
 * Authoritative onboarding state. Fresh until the ACN publishes a change for
 * it on `StreamChanges`; retained for the connection lifetime.
 */
export const GetOnboardingState = Acn.query("GetOnboardingState", {
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

export const CompleteOnboarding = Acn.mutation("CompleteOnboarding", {
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
