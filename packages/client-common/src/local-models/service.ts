import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Context, Effect, Layer, Option, Schema } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import { Models } from "../operations"
import {
  LocalModelMutationFailed,
  localModelManagedAcquisition,
  localModelResidency,
  type CustomFormModelId,
  type CustomLocalModel,
  type DrafterChoice,
  type HuggingFaceRepositoryId,
  type LocalModel,
  type ModelId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { localModelsFromCatalog } from "../model-catalog/projection"

export type LocalModelCommand = "install" | "cancel" | "dismiss" | "remove" | "load" | "drafter" | "hide" | "unhide"
export interface LocalModelCommandFailure {
  readonly operation: LocalModelCommand
  readonly rejection: Option.Option<LocalModelMutationFailed>
}
interface ModelCommandObservation {
  readonly modelId: ModelId
  readonly operation: LocalModelCommand
  readonly pending: boolean
  readonly failure: Option.Option<LocalModelCommandFailure>
}
export const localModelCommandFailure = (operation: LocalModelCommand, cause: Cause.Cause<unknown>): LocalModelCommandFailure => ({
  operation,
  rejection: Option.filter(Cause.failureOption(cause), Schema.is(LocalModelMutationFailed)),
})
export interface LocalModelStopStatus {
  readonly pending: boolean
  readonly failure: Option.Option<string>
}
export interface LocalModelCommandStatus {
  readonly pending: boolean
  readonly pendingOperations: ReadonlyArray<LocalModelCommand>
  readonly failures: ReadonlyArray<LocalModelCommandFailure>
}
/** Select exact model/command outcomes; a domain failure is displayed from its richer resource state. */
export const localModelCommandStatus = (modelId: ModelId, commands: ReadonlyArray<ReadonlyArray<ModelCommandObservation>>, model: Option.Option<LocalModel>): LocalModelCommandStatus => {
  const latest = commands.flatMap(history => {
    const observation = history.findLast(value => value.modelId === modelId)
    return observation ? [observation] : []
  })
  const superseded = (feedback: LocalModelCommandFailure): boolean => {
    if (Option.isNone(model)) return false
    const acquisition = Option.getOrUndefined(localModelManagedAcquisition(model.value))
    const residency = Option.getOrUndefined(localModelResidency(model.value))
    // Observed readiness fulfills a prior load request, regardless of who completed the load.
    if (feedback.operation === "load" && residency?._tag === "Ready") return true
    if (Option.isNone(feedback.rejection)) return false
    if (feedback.operation === "load" && residency?._tag === "Failed") return feedback.rejection.value.code === residency.failure.code
    if (feedback.operation === "remove" && acquisition?._tag === "RemoveFailed") return feedback.rejection.value.code === acquisition.failure.code
    return false
  }
  const pendingOperations = latest.filter(value => value.pending).map(value => value.operation)
  return { pending: pendingOperations.length > 0, pendingOperations,
    failures: latest.flatMap(value => value.pending ? [] : Option.toArray(value.failure)).filter(failure => !superseded(failure)) }
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
  const setDrafter = effectQuery.Models.SetModelDrafter
  const hide = effectQuery.Models.HideModel
  const unhide = effectQuery.Models.UnhideModel
  const addFromHuggingFace = effectQuery.Models.AddHuggingFaceModel
  const importFile = effectQuery.Models.ImportModelFile
  const stopStatus = Atom.make((get): LocalModelStopStatus => {
    const result = get(stop)
    return { pending: result.waiting, failure: Result.isFailure(result) && !result.waiting ? Option.some("The model may still be running. Try stopping it again.") : Option.none() }
  })
  const state = Atom.make((get) => Result.map(get(query).result, localModelsFromCatalog))
  const catalog = Atom.make((get) => Result.map(
    get(state),
    (models) => ({
      ...models,
      models: models.models.filter((model) => model._tag === "Catalog"),
    }),
  ))
  const custom = Atom.make((get) => Result.map(
    get(state),
    (models) => models.models.filter((model): model is CustomLocalModel => model._tag === "Custom"),
  ))
  const unsupported = Atom.make((get) => Result.map(get(state), (models) => models.unsupported))
  const hidden = Atom.make((get) => Result.map(get(state), (models) => models.hidden))
  const observeCommand = (mutation: typeof Models.LoadLocalModel | typeof Models.RemoveLocalModel | typeof Models.SyncLocalModel | typeof Models.CancelLocalModelSync | typeof Models.AcknowledgeLocalModelSyncFailure | typeof Models.SetModelDrafter | typeof Models.HideModel | typeof Models.UnhideModel, operation: LocalModelCommand) =>
    Mutation.state({ filters: { mutation }, select: ({ input, result }: {
      readonly input: { readonly modelId: ModelId }
      readonly result: Result.Result<unknown, unknown>
    }): ModelCommandObservation => ({
      modelId: input.modelId, operation, pending: result.waiting,
      failure: Result.isFailure(result) && !result.waiting ? Option.some(localModelCommandFailure(operation, result.cause)) : Option.none(),
    }) })
  const commandObservations = yield* Effect.all([
    observeCommand(Models.SyncLocalModel, "install"), observeCommand(Models.CancelLocalModelSync, "cancel"),
    observeCommand(Models.AcknowledgeLocalModelSyncFailure, "dismiss"), observeCommand(Models.RemoveLocalModel, "remove"),
    observeCommand(Models.LoadLocalModel, "load"), observeCommand(Models.SetModelDrafter, "drafter"),
    observeCommand(Models.HideModel, "hide"), observeCommand(Models.UnhideModel, "unhide"),
  ])
  const commandStatus = Atom.family((modelId: ModelId) => Atom.make(get => {
    const observed = get(state)
    const model = Result.isSuccess(observed) ? Option.fromNullable(observed.value.models.find(value => value.modelId === modelId)) : Option.none<LocalModel>()
    return localModelCommandStatus(modelId, commandObservations.map(observation => get(observation)), model)
  }))
  const provideRegistry = Effect.provideService(Registry.AtomRegistry, registry)

  return {
    state,
    catalog,
    custom,
    unsupported,
    hidden,
    commandStatus,
    stopStatus,
    load: (modelId: ModelId) => Mutation.execute(load, { modelId }).pipe(provideRegistry),
    stop: Mutation.execute(stop, {}).pipe(provideRegistry),
    retry: queryClient.invalidate(Models.GetCatalog.match()),
    install: (modelId: ModelId) => Mutation.execute(install, { modelId }).pipe(provideRegistry),
    cancelDownload: (modelId: ModelId) => Mutation.execute(cancelDownload, { modelId }).pipe(provideRegistry),
    dismissDownloadFailure: (modelId: ModelId) => Mutation.execute(dismissDownloadFailure, { modelId }).pipe(provideRegistry),
    remove: (modelId: ModelId) => Mutation.execute(remove, { modelId }).pipe(provideRegistry),
    setDrafter: (modelId: CustomFormModelId, drafter: DrafterChoice) =>
      Mutation.execute(setDrafter, { modelId, drafter }).pipe(provideRegistry),
    hide: (modelId: ModelId) => Mutation.execute(hide, { modelId }).pipe(provideRegistry),
    unhide: (modelId: ModelId) => Mutation.execute(unhide, { modelId }).pipe(provideRegistry),
    addFromHuggingFace: (input: {
      readonly repository: HuggingFaceRepositoryId
      readonly revision: Option.Option<string>
      readonly version: Option.Option<string>
      readonly drafter: DrafterChoice
      readonly force: boolean
    }) => Mutation.execute(addFromHuggingFace, input).pipe(provideRegistry),
    importFile: (path: string, drafter: DrafterChoice) =>
      Mutation.execute(importFile, { path, drafter }).pipe(provideRegistry),
  }
})

export interface LocalModels extends Effect.Effect.Success<typeof makeLocalModels> {}

export type LocalModelsInstallError = Effect.Effect.Error<ReturnType<LocalModels["install"]>>
export type LocalModelsCancelError = Effect.Effect.Error<ReturnType<LocalModels["cancelDownload"]>>

export const LocalModels = Context.GenericTag<LocalModels>("client/LocalModels")

export const LocalModelsLive = Layer.scoped(LocalModels, makeLocalModels)

export function useLocalModelMutations() {
  const client = useAgentClient()
  const install = useAtomSet(useMemo(() => client.runtime.fn((modelId: ModelId) => Effect.flatMap(LocalModels, models => models.install(modelId)), { concurrent: true }), [client]))
  const cancel = useAtomSet(useMemo(() => client.runtime.fn((modelId: ModelId) => Effect.flatMap(LocalModels, models => models.cancelDownload(modelId)), { concurrent: true }), [client]))
  const dismissFailure = useAtomSet(useMemo(() => client.runtime.fn((modelId: ModelId) => Effect.flatMap(LocalModels, models => models.dismissDownloadFailure(modelId)), { concurrent: true }), [client]))
  const remove = useAtomSet(useMemo(() => client.runtime.fn((modelId: ModelId) => Effect.flatMap(LocalModels, models => models.remove(modelId)), { concurrent: true }), [client]))
  const load = useAtomSet(useMemo(() => client.runtime.fn((modelId: ModelId) => Effect.flatMap(LocalModels, models => models.load(modelId)), { concurrent: true }), [client]))
  const stop = useAtomSet(useMemo(() => client.runtime.fn(() => Effect.flatMap(LocalModels, models => models.stop), { concurrent: true }), [client]))
  const setDrafter = useAtomSet(useMemo(() => client.runtime.fn(({ modelId, drafter }: { readonly modelId: CustomFormModelId; readonly drafter: DrafterChoice }) => Effect.flatMap(LocalModels, models => models.setDrafter(modelId, drafter)), { concurrent: true }), [client]))
  const hide = useAtomSet(useMemo(() => client.runtime.fn((modelId: ModelId) => Effect.flatMap(LocalModels, models => models.hide(modelId)), { concurrent: true }), [client]))
  const unhide = useAtomSet(useMemo(() => client.runtime.fn((modelId: ModelId) => Effect.flatMap(LocalModels, models => models.unhide(modelId)), { concurrent: true }), [client]))
  return {
    install: useCallback((modelId: ModelId) => install(modelId), [install]),
    cancel: useCallback((modelId: ModelId) => cancel(modelId), [cancel]),
    dismissFailure: useCallback((modelId: ModelId) => dismissFailure(modelId), [dismissFailure]),
    remove: useCallback((modelId: ModelId) => remove(modelId), [remove]),
    load: useCallback((modelId: ModelId) => load(modelId), [load]),
    stop: useCallback(() => stop(), [stop]),
    setDrafter: useCallback((modelId: CustomFormModelId, drafter: DrafterChoice) => setDrafter({ modelId, drafter }), [setDrafter]),
    hide: useCallback((modelId: ModelId) => hide(modelId), [hide]),
    unhide: useCallback((modelId: ModelId) => unhide(modelId), [unhide]),
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
  return Result.isSuccess(result) ? result.value : { pending: false, pendingOperations: [], failures: [] }
}
