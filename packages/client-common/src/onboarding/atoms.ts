import { Atom } from "@effect-atom/atom-react"
import { Data, Effect } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { AcnRpcClientTag, OnboardingMirror } from "@magnitudedev/sdk"
import type { AgentClientInstance } from "../state/agent-client"
import {
  getMirroredStateInvalidationWatch,
  subscribeToMirroredStateInvalidation,
} from "../hooks/use-mirrored-state"

export interface OnboardingUpdateInput {
  readonly completed: boolean
}

export class OnboardingSynchronizationFailed extends Data.TaggedError(
  "OnboardingSynchronizationFailed",
)<{
  readonly expectedCompleted: boolean
}> {}

export const onboardingQuery = Query.make("Onboarding", {
  key: (_: void) => Data.tuple("onboarding"),
  staleTime: Infinity,
  gcTime: Infinity,
  effect: () => Effect.flatMap(AcnRpcClientTag, (rpc) =>
    rpc("GetOnboardingState", {}).pipe(Effect.map(({ state }) => state))),
})

const synchronizeOnboarding = () => QueryClient.invalidate(onboardingQuery.match()).pipe(
  Effect.zipRight(QueryClient.fetch(onboardingQuery, undefined)),
)

export const updateOnboardingMutation = Mutation.make("UpdateOnboarding", {
  effect: ({ completed }: OnboardingUpdateInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("UpdateOnboardingState", { completed })),
  synchronize: (_, { completed }) => synchronizeOnboarding().pipe(
    Effect.filterOrFail(
      (state) => state.completed === completed,
      () => new OnboardingSynchronizationFailed({ expectedCompleted: completed }),
    ),
    Effect.asVoid,
  ),
})

const makeAtoms = (client: AgentClientInstance) => {
  const queryAtom = client.effectQuery.query(onboardingQuery, undefined)
  const updateMutation = client.effectQuery.mutation(updateOnboardingMutation)
  const invalidationBridgeAtom = client.effectQuery.runtime.atom(Effect.gen(function* () {
    const queryClient = yield* QueryClient.QueryClient
    yield* Effect.acquireRelease(
      Effect.sync(() => subscribeToMirroredStateInvalidation(
        client,
        OnboardingMirror.id,
        () => queryClient.invalidate(onboardingQuery.match()),
      )),
      (unsubscribe) => Effect.sync(unsubscribe),
    )
    yield* queryClient.prefetch(queryAtom)
    return yield* Effect.never
  }))

  return {
    queryAtom,
    resultAtom: Atom.make((get) => get(queryAtom).result),
    updateMutation,
    invalidationBridgeAtom,
    mirrorInvalidationWatchAtom: getMirroredStateInvalidationWatch(client, OnboardingMirror.id),
  }
}

export type OnboardingAtoms = ReturnType<typeof makeAtoms>

const atomsByClient = new WeakMap<object, OnboardingAtoms>()

export const onboardingAtoms = (client: AgentClientInstance): OnboardingAtoms => {
  const existing = atomsByClient.get(client)
  if (existing !== undefined) return existing
  const atoms = makeAtoms(client)
  atomsByClient.set(client, atoms)
  return atoms
}
