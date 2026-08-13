import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer, Option } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import {
  AcnRpcClientTag,
  LocalModelsMirror,
  type ModelDownloadId,
  type LocalModelInstallationAdmission,
  type LocalModelsState,
  type ModelServingConfigurationId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { EffectQueryInvalidations } from "../state/effect-query-invalidations"
import { findLocalModelByConfigurationId, localModelProviderModelId } from "./projection"

export class LocalModelSynchronizationFailed extends Data.TaggedError(
  "LocalModelSynchronizationFailed",
)<{ readonly operation: "install" | "cancel"; readonly message: string }> {}

interface InstallationInput {
  readonly configurationId: ModelServingConfigurationId
}

interface DownloadInput {
  readonly downloadId: ModelDownloadId
}

interface DeletionInput {
  readonly configurationId: ModelServingConfigurationId
}

const installationScope = (
  configurationId: ModelServingConfigurationId,
): Mutation.MutationScope => Mutation.MutationScope(configurationId)

const downloadScope = (
  downloadId: ModelDownloadId,
): Mutation.MutationScope => Mutation.MutationScope(downloadId)

const localModelsQuery = Query.make("LocalModels", {
  key: (_: void) => Data.tuple(LocalModelsMirror.id),
  staleTime: Infinity,
  gcTime: Infinity,
  effect: () => Effect.flatMap(AcnRpcClientTag, (rpc) =>
    rpc("GetLocalModels", {}).pipe(Effect.map(({ state }) => state))),
})

const synchronizeLocalModels = () => QueryClient.invalidate(localModelsQuery.match()).pipe(
  Effect.zipRight(QueryClient.fetch(localModelsQuery, undefined)),
)

export const installationAdmissionIsVisible = (
  state: LocalModelsState,
  configurationId: ModelServingConfigurationId,
  admission: LocalModelInstallationAdmission,
): boolean => Option.exists(
  findLocalModelByConfigurationId(state.models, configurationId),
  (model) => {
    const acquisition = model.acquisitionState
    if (acquisition._tag === "Installed") return true
    if (admission._tag === "AlreadyInstalled") return false
    return acquisition._tag !== "NotInstalled"
      && acquisition.downloadId === admission.downloadId
  },
)

const installMutation = Mutation.make("InstallModel", {
  scope: ({ configurationId }: InstallationInput) => installationScope(configurationId),
  effect: ({ configurationId }: InstallationInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("InstallModel", { configurationId })),
  synchronize: (admission, { configurationId }) => synchronizeLocalModels().pipe(
    Effect.filterOrFail(
      (state) => installationAdmissionIsVisible(state, configurationId, admission),
      () => new LocalModelSynchronizationFailed({
        operation: "install",
        message: "The admitted local-model installation was absent from LocalModels.",
      }),
    ),
    Effect.asVoid,
  ),
})

const cancelDownloadMutation = Mutation.make("CancelModelDownload", {
  scope: ({ downloadId }: DownloadInput) => downloadScope(downloadId),
  effect: ({ downloadId }: DownloadInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("CancelModelDownload", { downloadId })),
  synchronize: (_, { downloadId }) => synchronizeLocalModels().pipe(
    Effect.filterOrFail(
      (state) => state.models.every((model) => {
        const acquisition = model.acquisitionState
        return acquisition._tag === "NotInstalled"
          || acquisition._tag === "Installed"
          || acquisition.downloadId !== downloadId
          || acquisition._tag === "Cancelled"
          || acquisition._tag === "Failed"
      }),
      () => new LocalModelSynchronizationFailed({
        operation: "cancel",
        message: "The cancelled model download remained active in LocalModels.",
      }),
    ),
    Effect.asVoid,
  ),
})

const dismissDownloadFailureMutation = Mutation.make("DismissModelDownloadFailure", {
  scope: ({ downloadId }: DownloadInput) => downloadScope(downloadId),
  effect: ({ downloadId }: DownloadInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("DismissModelDownloadFailure", { downloadId })),
  synchronize: () => synchronizeLocalModels().pipe(Effect.asVoid),
})

const deleteModelMutation = Mutation.make("DeleteLocalModel", {
  scope: ({ configurationId }: DeletionInput) => installationScope(configurationId),
  effect: ({ configurationId }: DeletionInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("DeleteLocalModel", { configurationId })),
  synchronize: () => synchronizeLocalModels().pipe(Effect.asVoid),
})

const makeLocalModels = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const invalidations = yield* EffectQueryInvalidations
  const query = effectQuery.query(localModelsQuery, undefined)
  const install = effectQuery.mutation(installMutation)
  const cancelDownload = effectQuery.mutation(cancelDownloadMutation)
  const dismissDownloadFailure = effectQuery.mutation(dismissDownloadFailureMutation)
  const deleteModel = effectQuery.mutation(deleteModelMutation)
  const installationInvocations = Mutation.state({
    filters: { mutation: installMutation },
    select: ({ input, result }) => ({ configurationId: input.configurationId, result }),
  })
  const installationStatuses = Atom.make((get) => {
    const invocations = get(installationInvocations)
    const latestByConfigurationId = new Map(invocations.map(({ configurationId, result }) =>
      [configurationId, result] as const))
    const latest = invocations.at(-1)
    return {
      latestFailed: latest !== undefined && Result.isFailure(latest.result),
      installing: new Set([...latestByConfigurationId].flatMap(([configurationId, result]) =>
        Result.isWaiting(result) ? [configurationId] : [])),
      failed: new Set([...latestByConfigurationId].flatMap(([configurationId, result]) =>
        Result.isFailure(result) ? [configurationId] : [])),
    }
  })
  const invalidate = () => queryClient.invalidate(localModelsQuery.match())
  yield* invalidations.register(LocalModelsMirror.id, invalidate)
  const state = Atom.make((get) => get(query).result)

  return {
    state,
    installationStatuses,
    install: (configurationId: ModelServingConfigurationId) =>
      Mutation.execute(install, { configurationId }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    cancelDownload: (downloadId: ModelDownloadId) =>
      Mutation.execute(cancelDownload, { downloadId }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    dismissDownloadFailure: (downloadId: ModelDownloadId) =>
      Mutation.execute(dismissDownloadFailure, { downloadId }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    delete: (configurationId: ModelServingConfigurationId) =>
      Mutation.execute(deleteModel, { configurationId }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
  }
})

export interface LocalModels extends Effect.Effect.Success<typeof makeLocalModels> {}

export type LocalModelsInstallError = Effect.Effect.Error<ReturnType<LocalModels["install"]>>
export type LocalModelsCancelError = Effect.Effect.Error<ReturnType<LocalModels["cancelDownload"]>>

export const LocalModels = Context.GenericTag<LocalModels>("client/LocalModels")

export const LocalModelsLive = Layer.scoped(LocalModels, makeLocalModels)

export function useLocalModelMutations() {
  const client = useAgentClient()
  const service = useMemo(() => client.effectQuery.runtime.atom(LocalModels), [client])
  const statusesAtom = useMemo(() => Atom.make((get) =>
    Result.map(get(service), (localModels) => get(localModels.installationStatuses))), [service])
  const installAction = useMemo(() => client.effectQuery.runtime.fn<ModelServingConfigurationId>()(
    (configurationId) => Effect.flatMap(LocalModels, (models) => models.install(configurationId)),
  ), [client])
  const downloadAction = useMemo(() => client.effectQuery.runtime.fn<{
    readonly operation: "cancel" | "dismiss"
    readonly downloadId: ModelDownloadId
  }>()(({ operation, downloadId }) => Effect.flatMap(
    LocalModels,
    (models) => operation === "cancel"
      ? models.cancelDownload(downloadId)
      : models.dismissDownloadFailure(downloadId),
  )), [client])
  const deleteAction = useMemo(() => client.effectQuery.runtime.fn<ModelServingConfigurationId>()(
    (configurationId) => Effect.flatMap(LocalModels, (models) => models.delete(configurationId)),
  ), [client])
  const statusesResult = useAtomValue(statusesAtom)
  const statuses = Option.getOrElse(Result.value(statusesResult), () => ({
    latestFailed: false,
    installing: new Set<ModelServingConfigurationId>(),
    failed: new Set<ModelServingConfigurationId>(),
  }))
  const install = useAtomSet(installAction)
  const download = useAtomSet(downloadAction)
  const deleteModel = useAtomSet(deleteAction)

  return {
    isInstalling: (configurationId: ModelServingConfigurationId) =>
      statuses.installing.has(configurationId),
    installationFailed: (configurationId: ModelServingConfigurationId) =>
      statuses.failed.has(configurationId),
    latestInstallationFailed: statuses.latestFailed,
    install: useCallback((configurationId: ModelServingConfigurationId) => {
      install(configurationId)
    }, [install]),
    cancel: useCallback((downloadId: ModelDownloadId) => {
      download({ operation: "cancel", downloadId })
    }, [download]),
    dismissFailure: useCallback((downloadId: ModelDownloadId) => {
      download({ operation: "dismiss", downloadId })
    }, [download]),
    delete: useCallback((configurationId: ModelServingConfigurationId) => {
      deleteModel(configurationId)
    }, [deleteModel]),
  }
}
