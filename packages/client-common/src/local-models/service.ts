import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import {
  LocalInference,
  LocalModelSynchronizationFailed,
  findLocalModelByConfigurationId,
  localModelCatalogIdentity,
  localModelConfigurationId,
  synchronizeLocalModels,
  type CatalogIdentity,
  type ModelDownloadId,
  type LocalModelsState,
  type LocalModel,
  type ModelServingConfigurationId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"

export { LocalModelSynchronizationFailed, installationAdmissionIsVisible } from "@magnitudedev/sdk"

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
  readonly identity: CatalogIdentity
  readonly waiting: boolean
  readonly failed: boolean
}

interface DeletionInvocationState {
  readonly configurationId: ModelServingConfigurationId
  readonly waiting: boolean
  readonly failed: boolean
}

const sameCatalogIdentity = (left: CatalogIdentity, right: CatalogIdentity): boolean =>
  left.modelId === right.modelId && left.variantId === right.variantId

export const projectCatalogModelsView = (
  state: LocalModelsState,
  invocations: readonly ReconciliationInvocationState[],
  deletions: readonly DeletionInvocationState[] = [],
): CatalogModelsView => {
  const latestDeletionByConfigurationId = new Map(deletions.map((deletion) =>
    [deletion.configurationId, deletion] as const))
  return {
    discoveryState: state.discoveryState,
    models: state.models.flatMap((model): readonly CatalogModelView[] => {
      if (model.catalogMembershipState._tag !== "InCatalog") return []
      const acquisition = model.acquisitionState
      const upgrade = model.upgradeState
      const identity = localModelCatalogIdentity(model)
      const invocation = Option.isSome(identity)
        ? invocations.findLast((candidate) => sameCatalogIdentity(candidate.identity, identity.value))
        : undefined
      const configurationId = Option.getOrUndefined(localModelConfigurationId(model))
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

const makeLocalModels = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const query = effectQuery.LocalInference.GetLocalModels({})
  const install = effectQuery.LocalInference.ReconcileCatalogModel
  const cancelDownload = effectQuery.LocalInference.CancelModelDownload
  const dismissDownloadFailure = effectQuery.LocalInference.DismissModelDownloadFailure
  const deleteModel = effectQuery.LocalInference.DeleteLocalModel
  const installationInvocations = yield* Mutation.state({
    filters: { mutation: LocalInference.ReconcileCatalogModel },
    select: ({ input, result }): ReconciliationInvocationState => ({
      identity: input,
      waiting: Result.isWaiting(result),
      failed: Result.isFailure(result),
    }),
  })
  const deletionInvocations = yield* Mutation.state({
    filters: { mutation: LocalInference.DeleteLocalModel },
    select: ({ input, result }): DeletionInvocationState => ({
      configurationId: input.configurationId,
      waiting: Result.isWaiting(result),
      failed: Result.isFailure(result),
    }),
  })
  const latestInstallationFailed = Atom.make((get) =>
    get(installationInvocations).at(-1)?.failed ?? false)
  const state = Atom.make((get) => Result.map(get(query).result, (snapshot) => snapshot.state))
  const catalog = Atom.make((get) => Result.map(
    get(state),
    (models) => projectCatalogModelsView(
      models,
      get(installationInvocations),
      get(deletionInvocations),
    ),
  ))
  const provideRegistry = Effect.provideService(Registry.AtomRegistry, registry)
  const provideQueryClient = Effect.provideService(QueryClient.QueryClient, queryClient)

  /** Resolve the catalog identity behind a configuration from fresh inventory, then reconcile it. */
  const installConfiguration = (configurationId: ModelServingConfigurationId) =>
    synchronizeLocalModels.pipe(
      provideQueryClient,
      Effect.flatMap((current) => findLocalModelByConfigurationId(current.models, configurationId)),
      Effect.flatMap((model) => localModelCatalogIdentity(model).pipe(
        Effect.mapError(() => new LocalModelSynchronizationFailed({
          operation: "install",
          message: "Only catalog models can be reconciled.",
        })),
      )),
      Effect.flatMap((identity) => Mutation.execute(install, identity)),
      provideRegistry,
    )

  return {
    state,
    catalog,
    latestInstallationFailed,
    retry: queryClient.invalidate(LocalInference.GetLocalModels.match()),
    install: installConfiguration,
    cancelDownload: (downloadId: ModelDownloadId) =>
      Mutation.execute(cancelDownload, { downloadId }).pipe(provideRegistry),
    dismissDownloadFailure: (downloadId: ModelDownloadId) =>
      Mutation.execute(dismissDownloadFailure, { downloadId }).pipe(provideRegistry),
    delete: (configurationId: ModelServingConfigurationId) =>
      Mutation.execute(deleteModel, { configurationId }).pipe(provideRegistry),
  }
})

export interface LocalModels extends Effect.Effect.Success<typeof makeLocalModels> {}

export type LocalModelsInstallError = Effect.Effect.Error<ReturnType<LocalModels["install"]>>
export type LocalModelsCancelError = Effect.Effect.Error<ReturnType<LocalModels["cancelDownload"]>>

export const LocalModels = Context.GenericTag<LocalModels>("client/LocalModels")

export const LocalModelsLive = Layer.scoped(LocalModels, makeLocalModels)

export function useLocalModelMutations() {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(LocalModels), [client])
  const latestFailureAtom = useMemo(() => Atom.make((get) =>
    Result.map(get(service), (localModels) => get(localModels.latestInstallationFailed))), [service])
  const installAction = useMemo(() => client.runtime.fn<ModelServingConfigurationId>()(
    (configurationId) => Effect.flatMap(LocalModels, (models) => models.install(configurationId)),
  ), [client])
  const downloadAction = useMemo(() => client.runtime.fn<{
    readonly operation: "cancel" | "dismiss"
    readonly downloadId: ModelDownloadId
  }>()(({ operation, downloadId }) => Effect.flatMap(
    LocalModels,
    (models) => operation === "cancel"
      ? models.cancelDownload(downloadId)
      : models.dismissDownloadFailure(downloadId),
  )), [client])
  const deleteAction = useMemo(() => client.runtime.fn<ModelServingConfigurationId>()(
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
