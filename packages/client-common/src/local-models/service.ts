import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer, Option, Schedule } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import {
  AcnRpcClientTag,
  LocalModelsMirror,
  type ModelDownloadId,
  type CatalogModelReconciliationAdmission,
  type LocalModelsState,
  type LocalModel,
  type ModelServingConfigurationId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { EffectQueryInvalidations } from "../state/effect-query-invalidations"
import {
  findLocalModelByConfigurationId,
  localModelConfigurationId,
  localModelProviderModelId,
} from "./projection"

export class LocalModelSynchronizationFailed extends Data.TaggedError(
  "LocalModelSynchronizationFailed",
)<{ readonly operation: "install" | "cancel" | "delete"; readonly message: string }> {}

interface InstallationInput {
  readonly configurationId: ModelServingConfigurationId
}

interface DownloadInput {
  readonly downloadId: ModelDownloadId
}

interface DeletionInput {
  readonly configurationId: ModelServingConfigurationId
}

export type CatalogModelReconciliationKind = "Install" | "Update"

export type CatalogModelReconciliationState =
  | { readonly _tag: "Idle" }
  | { readonly _tag: "Starting"; readonly operation: CatalogModelReconciliationKind }
  | ({ readonly _tag: "Transferring"; readonly operation: CatalogModelReconciliationKind }
    & Omit<Extract<LocalModel["acquisitionState"], { readonly _tag: "Downloading" }>, "_tag">)
  | { readonly _tag: "Failed"; readonly operation: CatalogModelReconciliationKind }
  | { readonly _tag: "Removing" }
  | { readonly _tag: "RemoveFailed" }

export interface CatalogModelView {
  readonly model: LocalModel
  readonly reconciliationState: CatalogModelReconciliationState
}

export interface CatalogModelsView {
  readonly models: readonly CatalogModelView[]
  readonly discoveryState: LocalModelsState["discoveryState"]
}

interface ReconciliationInvocationState {
  readonly configurationId: ModelServingConfigurationId
  readonly waiting: boolean
  readonly failed: boolean
}

interface DeletionInvocationState {
  readonly configurationId: ModelServingConfigurationId
  readonly waiting: boolean
  readonly failed: boolean
}

export const projectCatalogModelsView = (
  state: LocalModelsState,
  invocations: readonly ReconciliationInvocationState[],
  deletions: readonly DeletionInvocationState[] = [],
): CatalogModelsView => {
  const latestByConfigurationId = new Map(invocations.map((invocation) =>
    [invocation.configurationId, invocation] as const))
  const latestDeletionByConfigurationId = new Map(deletions.map((deletion) =>
    [deletion.configurationId, deletion] as const))
  return {
    discoveryState: state.discoveryState,
    models: state.models.flatMap((model): readonly CatalogModelView[] => {
      if (model.catalogMembershipState._tag !== "InCatalog") return []
      const acquisition = model.acquisitionState
      const upgrade = model.upgradeState
      const configurationId = Option.getOrUndefined(localModelConfigurationId(model))
      const invocation = configurationId === undefined
        ? undefined
        : latestByConfigurationId.get(configurationId)
      const deletion = configurationId === undefined
        ? undefined
        : latestDeletionByConfigurationId.get(configurationId)
      const reconciliationState: CatalogModelReconciliationState = deletion?.waiting
        ? { _tag: "Removing" }
        : acquisition._tag === "Downloading"
        ? { ...acquisition, _tag: "Transferring", operation: "Install" }
        : upgrade._tag === "Upgrading"
          ? { ...upgrade, _tag: "Transferring", operation: "Update" }
          : invocation?.waiting
            ? {
                _tag: "Starting",
                operation: acquisition._tag === "Installed" ? "Update" : "Install",
              }
            : acquisition._tag === "Failed"
              ? { _tag: "Failed", operation: "Install" }
              : upgrade._tag === "Failed"
                ? { _tag: "Failed", operation: "Update" }
                : invocation?.failed
                  ? {
                      _tag: "Failed",
                      operation: acquisition._tag === "Installed" ? "Update" : "Install",
                    }
                  : deletion?.failed
                    ? { _tag: "RemoveFailed" }
                    : { _tag: "Idle" }
      return [{ model, reconciliationState }]
    }),
  }
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

const localModelsSynchronizationSchedule = Schedule.spaced("50 millis").pipe(
  Schedule.intersect(Schedule.recurs(100)),
)

const synchronizeLocalModelsUntil = (
  predicate: (state: LocalModelsState) => boolean,
  error: () => LocalModelSynchronizationFailed,
) => synchronizeLocalModels().pipe(
  Effect.filterOrFail(predicate, error),
  Effect.retry(localModelsSynchronizationSchedule),
)

export const installationAdmissionIsVisible = (
  state: LocalModelsState,
  configurationId: ModelServingConfigurationId,
  admission: CatalogModelReconciliationAdmission,
): boolean => Option.exists(
  findLocalModelByConfigurationId(state.models, configurationId),
  (model) => {
    const acquisition = model.acquisitionState
    const isCurrent = acquisition._tag === "Installed" && model.upgradeState._tag === "Current"
    if (admission._tag === "Current" || isCurrent) return isCurrent
    if (acquisition._tag !== "NotInstalled"
      && acquisition._tag !== "Installed"
      && acquisition.downloadId === admission.downloadId) return true
    return model.upgradeState._tag === "Upgrading"
      && model.upgradeState.downloadId === admission.downloadId
  },
)

const installMutation = Mutation.make("ReconcileCatalogModel", {
  scope: ({ configurationId }: InstallationInput) => installationScope(configurationId),
  effect: ({ configurationId }: InstallationInput) => synchronizeLocalModels().pipe(
    Effect.flatMap((state) => findLocalModelByConfigurationId(state.models, configurationId)),
    Effect.filterOrFail(
      (model) => model.catalogMembershipState._tag === "InCatalog",
      () => new LocalModelSynchronizationFailed({
        operation: "install",
        message: "Only catalog models can be reconciled.",
      }),
    ),
    Effect.flatMap((model) => {
      const catalog = model.catalogMembershipState
      if (catalog._tag !== "InCatalog") return Effect.die("Catalog membership changed")
      return Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("ReconcileCatalogModel", {
        modelId: catalog.catalogData.modelId,
        variantId: catalog.catalogData.variantId,
      }))
    }),
  ),
  synchronize: (admission, { configurationId }) => synchronizeLocalModelsUntil(
      (state) => installationAdmissionIsVisible(state, configurationId, admission),
      () => new LocalModelSynchronizationFailed({
        operation: "install",
        message: "The admitted local-model installation was absent from LocalModels.",
      }),
    ).pipe(
    Effect.asVoid,
  ),
})

const cancelDownloadMutation = Mutation.make("CancelModelDownload", {
  scope: ({ downloadId }: DownloadInput) => downloadScope(downloadId),
  effect: ({ downloadId }: DownloadInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("CancelModelDownload", { downloadId })),
  synchronize: (_, { downloadId }) => synchronizeLocalModelsUntil(
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
    ).pipe(
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
  synchronize: (_, { configurationId }) => synchronizeLocalModelsUntil(
    (state) => Option.match(
      findLocalModelByConfigurationId(state.models, configurationId),
      {
        onNone: () => true,
        onSome: (model) => model.acquisitionState._tag !== "Installed",
      },
    ),
    () => new LocalModelSynchronizationFailed({
      operation: "delete",
      message: "The deleted local model remained installed in LocalModels.",
    }),
  ).pipe(Effect.asVoid),
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
    select: ({ input, result }): ReconciliationInvocationState => ({
      configurationId: input.configurationId,
      waiting: Result.isWaiting(result),
      failed: Result.isFailure(result),
    }),
  })
  const deletionInvocations = Mutation.state({
    filters: { mutation: deleteModelMutation },
    select: ({ input, result }): DeletionInvocationState => ({
      configurationId: input.configurationId,
      waiting: Result.isWaiting(result),
      failed: Result.isFailure(result),
    }),
  })
  const latestInstallationFailed = Atom.make((get) =>
    get(installationInvocations).at(-1)?.failed ?? false)
  const invalidate = () => queryClient.invalidate(localModelsQuery.match())
  yield* invalidations.register(LocalModelsMirror.id, invalidate)
  const state = Atom.make((get) => get(query).result)
  const catalog = Atom.make((get) => Result.map(
    get(state),
    (models) => projectCatalogModelsView(
      models,
      get(installationInvocations),
      get(deletionInvocations),
    ),
  ))

  return {
    state,
    catalog,
    latestInstallationFailed,
    retry: queryClient.invalidate(localModelsQuery.match()),
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
  const latestFailureAtom = useMemo(() => Atom.make((get) =>
    Result.map(get(service), (localModels) => get(localModels.latestInstallationFailed))), [service])
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
  const latestFailureResult = useAtomValue(latestFailureAtom)
  const latestInstallationFailed = Option.getOrElse(Result.value(latestFailureResult), () => false)
  const install = useAtomSet(installAction)
  const download = useAtomSet(downloadAction)
  const deleteModel = useAtomSet(deleteAction)

  return {
    latestInstallationFailed,
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
