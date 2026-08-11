import { Atom, Result } from "@effect-atom/atom-react"
import { Data, Effect, Layer } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import {
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

const makeDefinitions = (client: AgentClientInstance) => {
  const runtime = Atom.runtime(Layer.merge(client.layer, QueryClient.layer))
  const query = Query.bind(runtime)
  const mutation = Mutation.bind(runtime)
  const localModelsQuery = query.make("LocalModels", {
    key: (_: void) => Data.tuple("local-models"),
    staleTime: Infinity,
    gcTime: Infinity,
    effect: () => Effect.flatMap(client, (rpc) =>
      rpc("GetLocalModels", {}).pipe(Effect.map(({ state }) => state))),
  })
  const synchronizeLocalModels = () => QueryClient.invalidate(
    localModelsQuery.match(),
    { refetch: false },
  ).pipe(
    Effect.zipRight(QueryClient.fetch(localModelsQuery(undefined))),
    Effect.asVoid,
  )

  const installMutation = mutation.make("InstallModel", {
    scope: ({ configurationId }: LocalModelInstallationInput) =>
      localModelInstallationScope(configurationId),
    effect: ({ configurationId }: LocalModelInstallationInput) =>
      Effect.flatMap(client, (rpc) => rpc("InstallModel", { configurationId })),
    synchronize: () => synchronizeLocalModels().pipe(
      Effect.zipRight(Reactivity.invalidate([ProviderModelCatalogMirror.id])),
    ),
  })
  const cancelDownloadMutation = mutation.make("CancelModelDownload", {
    scope: ({ attemptIds }: LocalModelDownloadInput) => localModelDownloadScope(attemptIds),
    effect: ({ attemptIds }: LocalModelDownloadInput) =>
      Effect.flatMap(client, (rpc) => rpc("CancelModelDownload", { attemptIds })),
    synchronize: synchronizeLocalModels,
  })
  const dismissDownloadFailureMutation = mutation.make("DismissModelDownloadFailure", {
    scope: ({ attemptIds }: LocalModelDownloadInput) => localModelDownloadScope(attemptIds),
    effect: ({ attemptIds }: LocalModelDownloadInput) =>
      Effect.flatMap(client, (rpc) => rpc("DismissModelDownloadFailure", { attemptIds })),
    synchronize: synchronizeLocalModels,
  })
  const deleteLocalModelMutation = mutation.make("DeleteLocalModel", {
    scope: ({ configurationId }: LocalModelDeletionInput) =>
      localModelInstallationScope(configurationId),
    effect: ({ configurationId }: LocalModelDeletionInput) =>
      Effect.flatMap(client, (rpc) => rpc("DeleteLocalModel", { configurationId })),
    synchronize: () => synchronizeLocalModels().pipe(
      Effect.zipRight(Reactivity.invalidate([
        ProviderModelCatalogMirror.id,
        ModelSlotsMirror.id,
      ])),
    ),
  })

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
    yield* queryClient.prefetch(localModelsQuery(undefined))
    return yield* Effect.never
  })
  const invalidationBridgeAtom = runtime.atom(invalidationBridgeEffect)
  const mirrorInvalidationWatchAtom = getMirroredStateInvalidationWatch(client, LocalModelsMirror.id)
  const localModelsResultAtom = Atom.make((get) => get(localModelsQuery(undefined)).result)
  const installationMutationStatesAtom = Mutation.state({
    filters: { mutation: installMutation },
  })

  return {
    localModelsQuery,
    localModelsResultAtom,
    installMutation,
    installationMutationStatesAtom,
    cancelDownloadMutation,
    dismissDownloadFailureMutation,
    deleteLocalModelMutation,
    invalidationBridgeAtom,
    mirrorInvalidationWatchAtom,
  }
}

export type LocalModelAtoms = ReturnType<typeof makeDefinitions>
export type LocalModelInstallationMutationState = Mutation.State<LocalModelAtoms["installMutation"]>

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

export const localModelAtoms = (
  client: AgentClientInstance,
): LocalModelAtoms => {
  const existing = atomsByClient.get(client)
  if (existing !== undefined) return existing
  const atoms = makeDefinitions(client)
  atomsByClient.set(client, atoms)
  return atoms
}
