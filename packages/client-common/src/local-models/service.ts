import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer, Option, Schema } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import {
  Configuration,
  CatalogIdentitySchema,
  Inference,
  findLocalModelById,
  localModelCatalogIdentity,
  ProviderModelIdSchema,
  type CatalogIdentity,
  ModelDownloadIdSchema,
  type ModelDownloadId,
  type LocalModelsState,
  type LocalModel,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"

export { installationAdmissionIsVisible } from "@magnitudedev/sdk"

export class LocalModelSynchronizationFailed extends Data.TaggedError(
  "LocalModelSynchronizationFailed",
)<{ readonly operation: "install" | "cancel" | "delete"; readonly message: string }> {}

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
  readonly identity: CatalogIdentity
  readonly waiting: boolean
  readonly failed: boolean
}

const sameCatalogIdentity = (left: CatalogIdentity, right: CatalogIdentity): boolean =>
  left.modelId === right.modelId && left.variantId === right.variantId

const catalogIdentityFromCanonicalModelId = (modelId: string): CatalogIdentity => {
  const separator = modelId.indexOf(":")
  if (separator <= 0 || separator === modelId.length - 1) {
    throw new TypeError(`Invalid canonical model ID: ${modelId}`)
  }
  return Schema.decodeUnknownSync(CatalogIdentitySchema)({
    modelId: modelId.slice(0, separator),
    variantId: modelId.slice(separator + 1),
  })
}

export const projectCatalogModelsView = (
  state: LocalModelsState,
  invocations: readonly ReconciliationInvocationState[],
  deletions: readonly DeletionInvocationState[] = [],
): CatalogModelsView => {
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
      const deletion = Option.isSome(identity)
        ? deletions.findLast((candidate) => sameCatalogIdentity(candidate.identity, identity.value))
        : undefined
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
  const query = effectQuery.Configuration.GetLocalModels({})
  const install = effectQuery.Inference.InstallInferenceModel
  const cancelDownload = effectQuery.Inference.CancelInferenceDownload
  const dismissDownloadFailure = effectQuery.Inference.AcknowledgeInferenceDownloadFailure
  const uninstall = effectQuery.Inference.UninstallInferenceModel
  const installationInvocations = yield* Mutation.state({
    filters: { mutation: Inference.InstallInferenceModel },
    select: ({ input, result }): ReconciliationInvocationState => ({
      identity: catalogIdentityFromCanonicalModelId(input.modelId),
      waiting: Result.isWaiting(result),
      failed: Result.isFailure(result),
    }),
  })
  const deletionInvocations = yield* Mutation.state({
    filters: { mutation: Inference.UninstallInferenceModel },
    select: ({ input, result }): DeletionInvocationState => ({
      identity: catalogIdentityFromCanonicalModelId(input.modelId),
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

  const currentModels = QueryClient.fetch(Configuration.GetLocalModels, {}).pipe(
    Effect.map(({ state }) => state),
  )
  const refreshLocalModels = queryClient.refetch(
    Configuration.GetLocalModels.match(),
  ).pipe(
    Effect.zipRight(QueryClient.fetch(Configuration.GetLocalModels, {})),
    Effect.asVoid,
    provideQueryClient,
  )

  const installModel = (modelId: ProviderModelId) =>
    currentModels.pipe(
      provideQueryClient,
      Effect.flatMap((current) => findLocalModelById(current.models, modelId)),
      Effect.flatMap((model) => localModelCatalogIdentity(model).pipe(
        Effect.mapError(() => new LocalModelSynchronizationFailed({
          operation: "install",
          message: "Only catalog models can be reconciled.",
        })),
      )),
      Effect.flatMap(() => Mutation.execute(install, { modelId }).pipe(
          Effect.map((admission) => admission._tag === "Current"
            ? {
                _tag: "Current" as const,
                providerModelId: ProviderModelIdSchema.make(modelId),
              }
            : {
                _tag: "DownloadAdmitted" as const,
                providerModelId: ProviderModelIdSchema.make(modelId),
                downloadId: ModelDownloadIdSchema.make(admission.downloadId),
              }),
        )),
      // The native mutation invalidates native inference resources. This
      // authored product service also consumes the ACN's richer local-model
      // projection, so request its refreshed projection as part of the
      // composition rather than coupling that concern into the inference
      // operation itself.
      Effect.tap(() => refreshLocalModels),
      provideRegistry,
    )

  const deleteModel = (modelId: ProviderModelId) =>
    currentModels.pipe(
      provideQueryClient,
      Effect.flatMap((current) => findLocalModelById(current.models, modelId)),
      Effect.flatMap((model) => localModelCatalogIdentity(model).pipe(
        Effect.mapError(() => new LocalModelSynchronizationFailed({
          operation: "delete",
          message: "Only catalog models can be uninstalled.",
        })),
      )),
      Effect.flatMap(() => Mutation.execute(uninstall, { modelId })),
      Effect.tap(() => refreshLocalModels),
      provideRegistry,
    )

  return {
    state,
    catalog,
    latestInstallationFailed,
    retry: queryClient.invalidate(Configuration.GetLocalModels.match()),
    install: installModel,
    cancelDownload: (downloadId: ModelDownloadId) =>
      Mutation.execute(cancelDownload, { downloadId }).pipe(
        Effect.tap(() => refreshLocalModels),
        provideRegistry,
      ),
    dismissDownloadFailure: (downloadId: ModelDownloadId) =>
      Mutation.execute(dismissDownloadFailure, { downloadId }).pipe(
        Effect.tap(() => refreshLocalModels),
        provideRegistry,
      ),
    delete: deleteModel,
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
  const installAction = useMemo(() => client.runtime.fn<ProviderModelId>()(
    (modelId) => Effect.flatMap(LocalModels, (models) => models.install(modelId)),
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
  const deleteAction = useMemo(() => client.runtime.fn<ProviderModelId>()(
    (modelId) => Effect.flatMap(LocalModels, (models) => models.delete(modelId)),
  ), [client])
  const latestFailureResult = useAtomValue(latestFailureAtom)
  const latestInstallationFailed = Option.getOrElse(Result.value(latestFailureResult), () => false)
  const install = useAtomSet(installAction)
  const download = useAtomSet(downloadAction)
  const deleteModel = useAtomSet(deleteAction)

  return {
    latestInstallationFailed,
    install: useCallback((modelId: ProviderModelId) => {
      install(modelId)
    }, [install]),
    cancel: useCallback((downloadId: ModelDownloadId) => {
      download({ operation: "cancel", downloadId })
    }, [download]),
    dismissFailure: useCallback((downloadId: ModelDownloadId) => {
      download({ operation: "dismiss", downloadId })
    }, [download]),
    delete: useCallback((modelId: ProviderModelId) => {
      deleteModel(modelId)
    }, [deleteModel]),
  }
}
