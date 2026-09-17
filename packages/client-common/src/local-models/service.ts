import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Context, Effect, Layer, Option, Schema } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import { Models } from "../operations"
import { type CatalogFormModelId, type ModelId } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { localModelsFromCatalog } from "../model-catalog/projection"

interface ModelCommandObservation {
  readonly modelId: ModelId
  readonly pending: boolean
  readonly failure: Option.Option<string>
}
const messageSchema = Schema.Struct({ message: Schema.NonEmptyString })
export const localModelFailureMessage = (cause: Cause.Cause<unknown>, fallback = "The model command could not finish. Check Status and try again."): string => {
  const failure = Cause.failureOption(cause)
  if (Option.isSome(failure) && Schema.is(messageSchema)(failure.value)) return failure.value.message
  return fallback
}
export interface LocalModelStopStatus {
  readonly pending: boolean
  readonly failure: Option.Option<string>
}
export interface LocalModelCommandStatus {
  readonly pending: boolean
  readonly failures: ReadonlyArray<string>
}
/** Select each command's latest invocation for this exact model, inside the domain owner. */
export const localModelCommandStatus = (modelId: ModelId, commands: ReadonlyArray<ReadonlyArray<ModelCommandObservation>>): LocalModelCommandStatus => {
  const latest = commands.flatMap(history => {
    const observation = history.findLast(value => value.modelId === modelId)
    return observation ? [observation] : []
  })
  return { pending: latest.some(value => value.pending), failures: latest.flatMap(value => Option.toArray(value.failure)) }
}

const makeLocalModels = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const query = effectQuery.Models.GetCatalog({})
  const install = effectQuery.Models.SyncLocalModel
  const cancelDownload = effectQuery.Models.CancelLocalModelSync
  const dismissDownloadFailure = effectQuery.Models.AcknowledgeLocalModelSyncFailure
  const remove = effectQuery.Models.RemoveLocalModel
  const load = effectQuery.Models.LoadLocalModel
  const stop = effectQuery.Models.StopActiveLocalModel
  const stopStatus = Atom.make((get): LocalModelStopStatus => {
    const result = get(stop)
    return { pending: result.waiting, failure: Result.isFailure(result) ? Option.some(localModelFailureMessage(result.cause)) : Option.none() }
  })
  const state = Atom.make((get) => Result.map(get(query).result, localModelsFromCatalog))
  const catalog = Atom.make((get) => Result.map(
    get(state),
    (models) => ({
      ...models,
      models: models.models.filter((model) => model._tag === "Catalog"),
    }),
  ))
  const commandObservations = yield* Effect.all([
    Models.SyncLocalModel, Models.CancelLocalModelSync, Models.AcknowledgeLocalModelSyncFailure,
    Models.RemoveLocalModel, Models.LoadLocalModel,
  ].map(mutation => Mutation.state({ filters: { mutation }, select: ({ input, result }): ModelCommandObservation => ({
    modelId: input.modelId, pending: result.waiting,
    failure: Result.isFailure(result) && !result.waiting ? Option.some(localModelFailureMessage(result.cause)) : Option.none(),
  }) })))
  const commandStatus = Atom.family((modelId: ModelId) => Atom.make(get =>
    localModelCommandStatus(modelId, commandObservations.map(observation => get(observation)))))
  const provideRegistry = Effect.provideService(Registry.AtomRegistry, registry)

  return {
    state,
    catalog,
    commandStatus,
    stopStatus,
    load: (modelId: ModelId) => Mutation.execute(load, { modelId }).pipe(provideRegistry),
    stop: Mutation.execute(stop, {}).pipe(provideRegistry),
    retry: queryClient.invalidate(Models.GetCatalog.match()),
    install: (modelId: CatalogFormModelId) => Mutation.execute(install, { modelId }).pipe(provideRegistry),
    cancelDownload: (modelId: CatalogFormModelId) =>
      Mutation.execute(cancelDownload, { modelId }).pipe(
        provideRegistry,
      ),
    dismissDownloadFailure: (modelId: CatalogFormModelId) =>
      Mutation.execute(dismissDownloadFailure, { modelId }).pipe(
        provideRegistry,
      ),
    remove: (modelId: CatalogFormModelId) => Mutation.execute(remove, { modelId }).pipe(provideRegistry),
  }
})

export interface LocalModels extends Effect.Effect.Success<typeof makeLocalModels> {}

export type LocalModelsInstallError = Effect.Effect.Error<ReturnType<LocalModels["install"]>>
export type LocalModelsCancelError = Effect.Effect.Error<ReturnType<LocalModels["cancelDownload"]>>

export const LocalModels = Context.GenericTag<LocalModels>("client/LocalModels")

export const LocalModelsLive = Layer.scoped(LocalModels, makeLocalModels)

export function useLocalModelMutations() {
  const client = useAgentClient()
  const install = useAtomSet(useMemo(() => client.runtime.fn((modelId: CatalogFormModelId) => Effect.flatMap(LocalModels, models => models.install(modelId)), { concurrent: true }), [client]))
  const cancel = useAtomSet(useMemo(() => client.runtime.fn((modelId: CatalogFormModelId) => Effect.flatMap(LocalModels, models => models.cancelDownload(modelId)), { concurrent: true }), [client]))
  const dismissFailure = useAtomSet(useMemo(() => client.runtime.fn((modelId: CatalogFormModelId) => Effect.flatMap(LocalModels, models => models.dismissDownloadFailure(modelId)), { concurrent: true }), [client]))
  const remove = useAtomSet(useMemo(() => client.runtime.fn((modelId: CatalogFormModelId) => Effect.flatMap(LocalModels, models => models.remove(modelId)), { concurrent: true }), [client]))
  const load = useAtomSet(useMemo(() => client.runtime.fn((modelId: ModelId) => Effect.flatMap(LocalModels, models => models.load(modelId)), { concurrent: true }), [client]))
  const stop = useAtomSet(useMemo(() => client.runtime.fn(() => Effect.flatMap(LocalModels, models => models.stop), { concurrent: true }), [client]))
  return {
    install: useCallback((modelId: CatalogFormModelId) => install(modelId), [install]),
    cancel: useCallback((modelId: CatalogFormModelId) => cancel(modelId), [cancel]),
    dismissFailure: useCallback((modelId: CatalogFormModelId) => dismissFailure(modelId), [dismissFailure]),
    remove: useCallback((modelId: CatalogFormModelId) => remove(modelId), [remove]),
    load: useCallback((modelId: ModelId) => load(modelId), [load]),
    stop: useCallback(() => stop(), [stop]),
  }
}

export function useLocalModelStopStatus(): LocalModelStopStatus {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(LocalModels), [client])
  const status = useMemo(() => Atom.make(get => Result.map(get(service), models => get(models.stopStatus))), [service])
  const result = useAtomValue(status)
  return Result.isSuccess(result) ? result.value : { pending: false, failure: Option.none() }
}

export function useLocalModelCommandStatus(modelId: ModelId): LocalModelCommandStatus {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(LocalModels), [client])
  const status = useMemo(() => Atom.make(get => Result.map(get(service), models => get(models.commandStatus(modelId)))), [service, modelId])
  const result = useAtomValue(status)
  return Result.isSuccess(result) ? result.value : { pending: false, failures: [] }
}
