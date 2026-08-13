import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { AcnRpcClientTag, OnboardingMirror } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { EffectQueryInvalidations } from "../state/effect-query-invalidations"

interface SetOnboardingCompletedInput {
  readonly completed: boolean
}

export class OnboardingPersistenceSynchronizationFailed extends Data.TaggedError(
  "OnboardingPersistenceSynchronizationFailed",
)<{
  readonly expectedCompleted: boolean
}> {}

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

const setOnboardingCompletedMutation = Mutation.make("UpdateOnboarding", {
  effect: ({ completed }: SetOnboardingCompletedInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("UpdateOnboardingState", { completed })),
  synchronize: (_, { completed }) => synchronizeOnboarding().pipe(
    Effect.filterOrFail(
      (state) => state.completed === completed,
      () => new OnboardingPersistenceSynchronizationFailed({ expectedCompleted: completed }),
    ),
    Effect.asVoid,
  ),
})

const makeOnboardingPersistence = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const invalidations = yield* EffectQueryInvalidations
  const query = effectQuery.query(onboardingQuery, undefined)
  const mutation = effectQuery.mutation(setOnboardingCompletedMutation)
  const updateResult = Atom.make((get) => get(mutation))
  const invalidate = () => queryClient.invalidate(onboardingQuery.match())

  yield* invalidations.register(OnboardingMirror.id, invalidate)

  return {
    state: Atom.make((get) => get(query).result),
    updateResult,
    setCompleted: (completed: boolean) => Mutation.execute(mutation, { completed }).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
    ),
  }
})

export interface OnboardingPersistence extends Effect.Effect.Success<typeof makeOnboardingPersistence> {}

export type OnboardingPersistenceError = Effect.Effect.Error<
  ReturnType<OnboardingPersistence["setCompleted"]>
>

export const OnboardingPersistence = Context.GenericTag<OnboardingPersistence>(
  "client/OnboardingPersistence",
)

export const OnboardingPersistenceLive = Layer.scoped(
  OnboardingPersistence,
  makeOnboardingPersistence,
)

export function useOnboardingState() {
  const client = useAgentClient()
  const service = useMemo(
    () => client.effectQuery.runtime.atom(OnboardingPersistence),
    [client],
  )
  const state = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (persistence) => get(persistence.state))), [service])
  const updateResult = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (persistence) => get(persistence.updateResult))), [service])
  const update = useMemo(() => client.effectQuery.runtime.fn<boolean>()(
    (completed) => Effect.flatMap(
      OnboardingPersistence,
      (persistence) => persistence.setCompleted(completed),
    ),
  ), [client])
  const setCompleted = useAtomSet(update)

  return {
    state: useAtomValue(state),
    updateResult: useAtomValue(updateResult),
    update: useCallback((completed: boolean) => setCompleted(completed), [setCompleted]),
  }
}
