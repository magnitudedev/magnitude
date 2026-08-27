import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer, Option } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import {
  Models,
  findLocalModelById,
  localModelCatalogIdentity,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { localModelsFromCatalog } from "../model-catalog/projection"

export class LocalModelSynchronizationFailed extends Data.TaggedError(
  "LocalModelSynchronizationFailed",
)<{ readonly operation: "install" | "cancel" | "remove"; readonly message: string }> {}

const makeLocalModels = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const query = effectQuery.Models.GetCatalog({})
  const install = effectQuery.Models.SyncLocalModel
  const cancelDownload = effectQuery.Models.CancelLocalModelSync
  const dismissDownloadFailure = effectQuery.Models.AcknowledgeLocalModelSyncFailure
  const remove = effectQuery.Models.RemoveLocalModel
  const installationInvocations = yield* Mutation.state({
    filters: { mutation: Models.SyncLocalModel },
    select: ({ result }) => Result.isFailure(result),
  })
  const latestInstallationFailed = Atom.make((get) =>
    get(installationInvocations).at(-1) ?? false)
  const state = Atom.make((get) => Result.map(
    get(query).result,
    localModelsFromCatalog,
  ))
  const catalog = Atom.make((get) => Result.map(
    get(state),
    (models) => ({
      ...models,
      models: models.models.filter((model) => model.catalogMembershipState._tag === "InCatalog"),
    }),
  ))
  const provideRegistry = Effect.provideService(Registry.AtomRegistry, registry)
  const provideQueryClient = Effect.provideService(QueryClient.QueryClient, queryClient)

  const currentModels = QueryClient.fetch(Models.GetCatalog, {}).pipe(Effect.map(localModelsFromCatalog))
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
      Effect.flatMap(() => Mutation.execute(install, { modelId })),
      provideRegistry,
    )

  const removeModel = (modelId: ProviderModelId) =>
    currentModels.pipe(
      provideQueryClient,
      Effect.flatMap((current) => findLocalModelById(current.models, modelId)),
      Effect.flatMap((model) => localModelCatalogIdentity(model).pipe(
        Effect.mapError(() => new LocalModelSynchronizationFailed({
          operation: "remove",
          message: "Only catalog models can be removed.",
        })),
      )),
      Effect.flatMap(() => Mutation.execute(remove, { modelId })),
      provideRegistry,
    )

  return {
    state,
    catalog,
    latestInstallationFailed,
    retry: queryClient.invalidate(Models.GetCatalog.match()),
    install: installModel,
    cancelDownload: (modelId: ProviderModelId) =>
      Mutation.execute(cancelDownload, { modelId }).pipe(
        provideRegistry,
      ),
    dismissDownloadFailure: (modelId: ProviderModelId) =>
      Mutation.execute(dismissDownloadFailure, { modelId }).pipe(
        provideRegistry,
      ),
    remove: removeModel,
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
    readonly modelId: ProviderModelId
  }>()(({ operation, modelId }) => Effect.flatMap(
    LocalModels,
    (models) => operation === "cancel"
      ? models.cancelDownload(modelId)
      : models.dismissDownloadFailure(modelId),
  )), [client])
  const removeAction = useMemo(() => client.runtime.fn<ProviderModelId>()(
    (modelId) => Effect.flatMap(LocalModels, (models) => models.remove(modelId)),
  ), [client])
  const latestFailureResult = useAtomValue(latestFailureAtom)
  const latestInstallationFailed = Option.getOrElse(Result.value(latestFailureResult), () => false)
  const install = useAtomSet(installAction)
  const download = useAtomSet(downloadAction)
  const removeModel = useAtomSet(removeAction)

  return {
    latestInstallationFailed,
    install: useCallback((modelId: ProviderModelId) => {
      install(modelId)
    }, [install]),
    cancel: useCallback((modelId: ProviderModelId) => {
      download({ operation: "cancel", modelId })
    }, [download]),
    dismissFailure: useCallback((modelId: ProviderModelId) => {
      download({ operation: "dismiss", modelId })
    }, [download]),
    remove: useCallback((modelId: ProviderModelId) => {
      removeModel(modelId)
    }, [removeModel]),
  }
}
