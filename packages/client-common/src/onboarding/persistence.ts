import { Atom, Registry } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer, Stream } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { AcnRpcClientTag, OnboardingMirror } from "@magnitudedev/sdk"
import { ClientEffectQuery } from "../state/client-effect-query"
import { ClientInvalidations } from "../state/client-invalidations"

export class OnboardingPersistenceSynchronizationFailed extends Data.TaggedError(
  "OnboardingPersistenceSynchronizationFailed",
)<{}> {}

const onboardingQuery = Query.make("Onboarding", {
  key: (_: void) => Data.tuple(OnboardingMirror.id),
  staleTime: Infinity,
  gcTime: Infinity,
  effect: () => Effect.flatMap(AcnRpcClientTag, (rpc) =>
    rpc("GetOnboardingState", {}).pipe(Effect.map(({ state }) => state))),
})

const synchronizeOnboarding = () => QueryClient.invalidate(onboardingQuery.match()).pipe(
  Effect.zipRight(QueryClient.fetch(onboardingQuery, undefined)),
)

const completeOnboardingMutation = Mutation.make("CompleteOnboarding", {
  effect: (_: void) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("CompleteOnboarding", {})),
  synchronize: () => synchronizeOnboarding().pipe(
    Effect.filterOrFail(
      (state) => state.completed,
      () => new OnboardingPersistenceSynchronizationFailed(),
    ),
    Effect.asVoid,
  ),
})

const makeOnboardingPersistence = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const invalidations = yield* ClientInvalidations
  const query = effectQuery.query(onboardingQuery, undefined)
  const mutation = effectQuery.mutation(completeOnboardingMutation)
  yield* invalidations.mirrors(OnboardingMirror.id).pipe(
    Stream.runForEach(() => queryClient.invalidate(onboardingQuery.match())),
    Effect.forkScoped,
  )

  return {
    state: Atom.make((get) => get(query).result),
    retry: queryClient.invalidate(onboardingQuery.match()),
    complete: Mutation.execute(mutation, undefined).pipe(
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
