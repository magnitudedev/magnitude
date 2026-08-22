import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Effect, Layer } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import { CompleteOnboarding, GetOnboardingState } from "@magnitudedev/sdk"
import { ClientEffectQuery } from "../state/client-effect-query"

export { OnboardingPersistenceSynchronizationFailed } from "@magnitudedev/sdk"

const makeOnboardingPersistence = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const query = effectQuery.query(GetOnboardingState, {})
  const complete = effectQuery.mutation(CompleteOnboarding)

  return {
    state: Atom.make((get) => Result.map(get(query).result, (snapshot) => snapshot.state)),
    retry: queryClient.invalidate(GetOnboardingState.match()),
    complete: Mutation.execute(complete, {}).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
    ),
  }
})

export interface OnboardingPersistence extends Effect.Effect.Success<typeof makeOnboardingPersistence> {}

export type OnboardingPersistenceError = Effect.Effect.Error<
  OnboardingPersistence["complete"]
>

export const OnboardingPersistence = Context.GenericTag<OnboardingPersistence>(
  "client/OnboardingPersistence",
)

export const OnboardingPersistenceLive = Layer.scoped(
  OnboardingPersistence,
  makeOnboardingPersistence,
)
