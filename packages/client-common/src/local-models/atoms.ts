import { Atom, Result } from "@effect-atom/atom-react"
import { Data, Effect } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import {
  AcnRpcClientTag,
  LocalModelsMirror,
  ModelSlotsMirror,
  ProviderModelCatalogMirror,
  type DownloadAttemptId,
  type ModelServingConfigurationId,
} from "@magnitudedev/sdk"
import * as Reactivity from "@effect/experimental/Reactivity"
import type { AgentClientInstance } from "../state/agent-client"
import {
  getMirroredStateInvalidationWatch,
  subscribeToMirroredStateInvalidation,
} from "../hooks/use-mirrored-state"

export interface LocalModelInstallationInput {
  readonly configurationId: ModelServingConfigurationId
}

export interface LocalModelDownloadInput {
  readonly attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]]
}

export interface LocalModelDeletionInput {
  readonly configurationId: ModelServingConfigurationId
}

export const localModelInstallationScope = (
  configurationId: ModelServingConfigurationId,
): Mutation.MutationScope => Mutation.MutationScope(configurationId)

const localModelDownloadScope = (
  attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]],
): Mutation.MutationScope => Mutation.MutationScope([...attemptIds].sort().join("|"))

export const localModelsQuery = Query.make("LocalModels", {
  key: (_: void) => Data.tuple("local-models"),
  staleTime: Infinity,
  gcTime: Infinity,
  effect: () => Effect.flatMap(AcnRpcClientTag, (rpc) =>
    rpc("GetLocalModels", {}).pipe(Effect.map(({ state }) => state))),
})

const synchronizeLocalModels = () => QueryClient.invalidate(
  localModelsQuery.match(),
).pipe(
  Effect.zipRight(QueryClient.fetch(localModelsQuery, undefined)),
  Effect.asVoid,
)

export const installLocalModelMutation = Mutation.make("InstallModel", {
  scope: ({ configurationId }: LocalModelInstallationInput) =>
    localModelInstallationScope(configurationId),
  effect: ({ configurationId }: LocalModelInstallationInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("InstallModel", { configurationId })),
  synchronize: () => synchronizeLocalModels().pipe(
    Effect.zipRight(Reactivity.invalidate([ProviderModelCatalogMirror.id])),
  ),
})

export const cancelModelDownloadMutation = Mutation.make("CancelModelDownload", {
  scope: ({ attemptIds }: LocalModelDownloadInput) => localModelDownloadScope(attemptIds),
  effect: ({ attemptIds }: LocalModelDownloadInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("CancelModelDownload", { attemptIds })),
  synchronize: synchronizeLocalModels,
})

export const dismissModelDownloadFailureMutation = Mutation.make("DismissModelDownloadFailure", {
  scope: ({ attemptIds }: LocalModelDownloadInput) => localModelDownloadScope(attemptIds),
  effect: ({ attemptIds }: LocalModelDownloadInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("DismissModelDownloadFailure", { attemptIds })),
  synchronize: synchronizeLocalModels,
})

export const deleteLocalModelMutation = Mutation.make("DeleteLocalModel", {
  scope: ({ configurationId }: LocalModelDeletionInput) =>
    localModelInstallationScope(configurationId),
  effect: ({ configurationId }: LocalModelDeletionInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("DeleteLocalModel", { configurationId })),
  synchronize: () => synchronizeLocalModels().pipe(
    Effect.zipRight(Reactivity.invalidate([
      ProviderModelCatalogMirror.id,
      ModelSlotsMirror.id,
    ])),
  ),
})

const makeAtoms = (client: AgentClientInstance) => {
  const effectQuery = client.effectQuery
  const localModelsQueryAtom = effectQuery.query(localModelsQuery, undefined)
  const installMutation = effectQuery.mutation(installLocalModelMutation)
  const cancelDownloadMutation = effectQuery.mutation(cancelModelDownloadMutation)
  const dismissDownloadFailureMutation = effectQuery.mutation(dismissModelDownloadFailureMutation)
  const deleteLocalModelMutationAtom = effectQuery.mutation(deleteLocalModelMutation)

  const invalidationBridgeEffect = Effect.gen(function* () {
    const queryClient = yield* QueryClient.QueryClient
    yield* Effect.acquireRelease(
      Effect.sync(() => subscribeToMirroredStateInvalidation(
        client,
        LocalModelsMirror.id,
        () => queryClient.invalidate(localModelsQuery.match()),
      )),
      (unsubscribe) => Effect.sync(unsubscribe),
    )
    yield* queryClient.prefetch(localModelsQueryAtom)
    return yield* Effect.never
  })

  return {
    localModelsQueryAtom,
    localModelsResultAtom: Atom.make((get) => get(localModelsQueryAtom).result),
    installMutation,
    installationMutationStatesAtom: Mutation.state({
      filters: { mutation: installLocalModelMutation },
    }),
    cancelDownloadMutation,
    dismissDownloadFailureMutation,
    deleteLocalModelMutation: deleteLocalModelMutationAtom,
    invalidationBridgeAtom: effectQuery.runtime.atom(invalidationBridgeEffect),
    mirrorInvalidationWatchAtom: getMirroredStateInvalidationWatch(client, LocalModelsMirror.id),
  }
}

export type LocalModelAtoms = ReturnType<typeof makeAtoms>
export type LocalModelInstallationMutationState = Mutation.State<typeof installLocalModelMutation>

export const localModelInstallationIsPending = (
  mutationStates: ReadonlyArray<LocalModelInstallationMutationState>,
  configurationId: ModelServingConfigurationId,
): boolean => mutationStates.some(({ input, result }) =>
  input.configurationId === configurationId && Result.isWaiting(result))

export const latestLocalModelInstallationMutationState = (
  mutationStates: ReadonlyArray<LocalModelInstallationMutationState>,
  configurationId: ModelServingConfigurationId,
): LocalModelInstallationMutationState | undefined => mutationStates.findLast(({ input }) =>
  input.configurationId === configurationId)

const atomsByClient = new WeakMap<object, LocalModelAtoms>()

export const localModelAtoms = (client: AgentClientInstance): LocalModelAtoms => {
  const existing = atomsByClient.get(client)
  if (existing !== undefined) return existing
  const atoms = makeAtoms(client)
  atomsByClient.set(client, atoms)
  return atoms
}
