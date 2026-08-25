import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer, Option } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import {
  Configuration,
  Inference,
  ModelSlotConfiguredLocal,
  ModelSlotConfiguredRemote,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  authoritativeSlotSelection,
  modelSlotActions,
  projectInferenceResidency,
  type ModelSlotsState,
  type ModelSlotSelectionsState,
  type ModelSlot,
  type ModelSlotAvailability,
  type ModelResidency,
  type InferenceModelsResponse,
  type InferenceInstancesSnapshot,
  type ProviderModelCatalogState,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"

export {
  ModelSlotSynchronizationFailed,
  authoritativeSlotSelection,
  modelLoadIsVisible,
  sameSlotSelection,
  selectedModelStopIsVisible,
  slotAssignmentIsVisible,
} from "@magnitudedev/sdk"

interface PendingAssignment {
  readonly slotId: SlotId
  readonly selection: SlotSelection
  readonly pending: boolean
}

export class ModelSlotControlRejected extends Data.TaggedError("ModelSlotControlRejected")<{
  readonly slotId: SlotId
  readonly message: string
}> {}

const unavailable = (code: string, message: string, retryable = true): ModelSlotAvailability => ({
  _tag: "Unavailable",
  failure: { code, message, retryable },
})

const projectSlotIntent = (
  intent: ModelSlotSelectionsState,
  catalog: ProviderModelCatalogState,
  nativeModels: InferenceModelsResponse,
): ModelSlotsState => {
  const providers = catalog._tag === "Loading" ? [] : catalog.providers
  const providerModels = catalog._tag === "Loading" || catalog._tag === "Unavailable"
    ? []
    : catalog.models
  const project = (slotId: SlotId, selection: Option.Option<SlotSelection>): ModelSlot =>
    Option.match(selection, {
      onNone: () => new ModelSlotUnassigned({ slotId }),
      onSome: (selected) => {
        const provider = providers.find(({ providerId }) => providerId === selected.providerId)
        const model = providerModels.find(({ providerId, providerModelId }) =>
          providerId === selected.providerId && providerModelId === selected.providerModelId)
        const descriptor = {
          providerId: selected.providerId,
          providerModelId: selected.providerModelId,
          displayName: model?.displayName || selected.providerModelId,
          variantLabel: model?.variantLabel ?? Option.none(),
        }
        let availability: ModelSlotAvailability
        if (catalog._tag === "Loading" || catalog._tag === "Refreshing" && model === undefined) {
          availability = { _tag: "Pending" }
        } else if (provider === undefined) {
          availability = unavailable("provider_unavailable", "The selected provider is unavailable")
        } else if (provider.authentication === "NotConfigured") {
          availability = unavailable("provider_not_configured", "The selected provider is not configured", false)
        } else if (provider.availability._tag !== "Available") {
          availability = unavailable("provider_unavailable", "The selected provider is unavailable")
        } else if (model === undefined || model.availability._tag !== "Available") {
          availability = unavailable("model_unavailable", "The selected model is unavailable")
        } else {
          availability = { _tag: "Available" }
        }
        if (selected.providerId !== "local") {
          return new ModelSlotConfiguredRemote({
            slotId,
            selection: selected,
            descriptor,
            availability,
            actions: [],
          })
        }
        const native = nativeModels.models.find((candidate) =>
          candidate.id === selected.providerModelId)
        if (availability._tag === "Available"
          && (native === undefined || native.localState._tag !== "Installed")) {
          availability = unavailable("model_not_installed", "The selected model is not installed")
        }
        const residency = { _tag: "Unloaded" as const }
        return new ModelSlotConfiguredLocal({
          slotId,
          selection: selected,
          descriptor,
          availability,
          residency,
          actions: modelSlotActions(availability, residency),
        })
      },
    })
  return {
    slots: {
      primary: project(PRIMARY_SLOT_ID, intent.slots.primary),
      secondary: project(SECONDARY_SLOT_ID, intent.slots.secondary),
    },
    recentModels: intent.recentModels,
    favoriteModels: intent.favoriteModels,
  }
}

const projectLocalSlotResidency = (
  slot: Extract<ModelSlot, { readonly _tag: "ConfiguredLocal" }>,
  instances: InferenceInstancesSnapshot,
): ModelResidency => {
  const instance = instances.instances.findLast((candidate) =>
    candidate.modelId === slot.selection.providerModelId)
  return instance === undefined ? { _tag: "Unloaded" } : projectInferenceResidency(instance)
}

const projectSlotResidency = (
  state: ModelSlotsState,
  instances: InferenceInstancesSnapshot,
): ModelSlotsState => {
  const project = (slot: ModelSlot): ModelSlot => {
    if (slot._tag !== "ConfiguredLocal") return slot
    const residency = projectLocalSlotResidency(slot, instances)
    return {
      ...slot,
      residency,
      actions: modelSlotActions(slot.availability, residency),
    }
  }
  return {
    ...state,
    slots: {
      primary: project(state.slots.primary),
      secondary: project(state.slots.secondary),
    },
  }
}

type Snapshot<State> = {
  readonly revision: number
  readonly state: State
}

/**
 * Compose the four independently refreshed model-state resources in dependency
 * order. `Result.all` cannot be used here: on failure it preserves the failed
 * member's `previousSuccess`, whose value is not the aggregate object.
 */
export const projectModelSlotsResult = <SlotError, CatalogError, ModelsError, InstancesError>(
  slotsResult: Result.Result<Snapshot<ModelSlotSelectionsState>, SlotError>,
  catalogResult: Result.Result<Snapshot<ProviderModelCatalogState>, CatalogError>,
  modelsResult: Result.Result<InferenceModelsResponse, ModelsError>,
  instancesResult: Result.Result<InferenceInstancesSnapshot, InstancesError>,
) => Result.flatMap(
  slotsResult,
  (slots) => Result.flatMap(
    catalogResult,
    (catalog) => Result.flatMap(
      modelsResult,
      (models) => Result.map(instancesResult, (instances) => ({
        ...slots,
        state: projectSlotResidency(
          projectSlotIntent(slots.state, catalog.state, models),
          instances,
        ),
      })),
    ),
  ),
)

/** The latest pending exact assignment for a slot, presented over authoritative state. */
export const presentedSlotSelection = (
  state: ModelSlotsState,
  assignments: ReadonlyArray<PendingAssignment>,
  slotId: SlotId,
): Option.Option<SlotSelection> => {
  const pending = assignments.findLast((assignment) =>
    assignment.slotId === slotId && assignment.pending)
  return pending === undefined
    ? authoritativeSlotSelection(state, slotId)
    : Option.some(pending.selection)
}

const makeModelSlots = Effect.gen(function* () {
  const effectQuery = yield* ClientEffectQuery
  const queryClient = yield* QueryClient.QueryClient
  const registry = yield* Registry.AtomRegistry
  const query = effectQuery.Configuration.GetModelSlots({})
  const catalogQuery = effectQuery.Configuration.GetProviderModelCatalog({})
  const modelsQuery = effectQuery.Inference.GetInferenceModels({})
  const instancesQuery = effectQuery.Inference.GetInferenceInstances({})
  const assign = effectQuery.Configuration.AssignSlot
  const clear = effectQuery.Configuration.ClearSlot
  const ensure = effectQuery.Inference.EnsureInferenceInstance
  const stopInstance = effectQuery.Inference.StopInferenceInstance
  const favorite = effectQuery.Configuration.SetModelFavorite
  const assignResult = Atom.make((get) => get(assign))
  const clearResult = Atom.make((get) => get(clear))
  const favoriteResult = Atom.make((get) => get(favorite))
  const assignments = yield* Mutation.state({
    filters: { mutation: Configuration.AssignSlot },
    select: ({ input, result }): PendingAssignment => ({
      slotId: input.slotId,
      selection: input.selection,
      pending: Result.isWaiting(result),
    }),
  })
  const state = Atom.make((get) => projectModelSlotsResult(
    get(query).result,
    get(catalogQuery).result,
    get(modelsQuery).result,
    get(instancesQuery).result,
  ))
  const selections = Atom.make((get) => Result.map(get(state), ({ state: current }) => ({
    primary: presentedSlotSelection(current, get(assignments), PRIMARY_SLOT_ID),
    secondary: presentedSlotSelection(
      current,
      get(assignments),
      SECONDARY_SLOT_ID,
    ),
  })))
  const provideRegistry = Effect.provideService(Registry.AtomRegistry, registry)
  const selectedLocalSlot = (slotId: SlotId) => QueryClient.fetch(
    Configuration.GetModelSlots,
    {},
  ).pipe(
    Effect.flatMap(({ state: current }) => Effect.all({
      catalog: QueryClient.fetch(Configuration.GetProviderModelCatalog, {}),
      models: QueryClient.fetch(Inference.GetInferenceModels, {}),
    }).pipe(Effect.map(({ catalog, models }) => projectSlotIntent(current, catalog.state, models)))),
    Effect.map((current) => slotId === PRIMARY_SLOT_ID ? current.slots.primary : current.slots.secondary),
    Effect.filterOrFail(
      (slot) => slot._tag === "ConfiguredLocal",
      () => new ModelSlotControlRejected({
        slotId,
        message: "The slot does not contain a local model",
      }),
    ),
    Effect.provideService(QueryClient.QueryClient, queryClient),
  )

  const selectedLocalResidency = (slotId: SlotId) => Effect.all({
    selected: selectedLocalSlot(slotId),
    instances: QueryClient.fetch(Inference.GetInferenceInstances, {}).pipe(
      Effect.provideService(QueryClient.QueryClient, queryClient),
    ),
  }).pipe(
    Effect.map(({ selected, instances }) => projectLocalSlotResidency(
      selected,
      instances,
    )),
  )

  return {
    state,
    selections,
    assignResult,
    clearResult,
    favoriteResult,
    retry: queryClient.invalidate(Configuration.GetModelSlots.match()),
    assign: (slotId: SlotId, selection: SlotSelection) =>
      Mutation.execute(assign, { slotId, selection }).pipe(provideRegistry),
    clear: (slotId: SlotId) => Mutation.execute(clear, { slotId }).pipe(provideRegistry),
    load: (slotId: SlotId) => selectedLocalSlot(slotId).pipe(
      Effect.flatMap((slot) => Mutation.execute(ensure, {
        modelId: slot.selection.providerModelId,
      })),
      provideRegistry,
    ),
    stop: (slotId: SlotId) => selectedLocalResidency(slotId).pipe(
      Effect.flatMap((residency) => Effect.gen(function* () {
        if (residency._tag === "Loading"
          || residency._tag === "Ready"
          || residency._tag === "Stopping") {
          return yield* Mutation.execute(stopInstance, { instanceId: residency.instanceId })
        }
        return yield* new ModelSlotControlRejected({
          slotId,
          message: "The selected model has no active instance",
        })
      })),
      provideRegistry,
    ),
    setFavorite: (model: ProviderModelIdentity, favoriteValue: boolean) =>
      Mutation.execute(favorite, { model, favorite: favoriteValue }).pipe(provideRegistry),
  }
})

export interface ModelSlots extends Effect.Effect.Success<typeof makeModelSlots> {}

export type ModelSlotsAssignError = Effect.Effect.Error<ReturnType<ModelSlots["assign"]>>
export type ModelSlotsLoadError = Effect.Effect.Error<ReturnType<ModelSlots["load"]>>
export type ModelSlotsStopError = Effect.Effect.Error<ReturnType<ModelSlots["stop"]>>

export const ModelSlots = Context.GenericTag<ModelSlots>("client/ModelSlots")

export const ModelSlotsLive = Layer.scoped(ModelSlots, makeModelSlots)

export function useModelSlotMutations() {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(ModelSlots), [client])
  const resultAtom = useMemo(() => Atom.make((get) => Result.map(get(service), (slots) => ({
    assign: get(slots.assignResult),
    clear: get(slots.clearResult),
    favorite: get(slots.favoriteResult),
  }))), [service])
  const action = useMemo(() => client.runtime.fn<
    | { readonly _tag: "Assign"; readonly slotId: SlotId; readonly selection: SlotSelection }
    | { readonly _tag: "Clear"; readonly slotId: SlotId }
    | { readonly _tag: "Favorite"; readonly model: ProviderModelIdentity; readonly favorite: boolean }
  >()((input) => Effect.flatMap(ModelSlots, (slots) => Effect.gen(function* () {
    switch (input._tag) {
      case "Assign": return yield* slots.assign(input.slotId, input.selection)
      case "Clear": return yield* slots.clear(input.slotId)
      case "Favorite": return yield* slots.setFavorite(input.model, input.favorite)
    }
  }))), [client])
  const controlAction = useMemo(() => client.runtime.fn<
    | { readonly _tag: "Load"; readonly slotId: SlotId }
    | { readonly _tag: "Stop"; readonly slotId: SlotId }
  >()((input) => Effect.flatMap(ModelSlots, (slots) => Effect.gen(function* () {
    switch (input._tag) {
      case "Load": return yield* slots.load(input.slotId).pipe(Effect.asVoid)
      case "Stop": return yield* slots.stop(input.slotId)
    }
  }))), [client])
  const results = useAtomValue(resultAtom)
  const invoke = useAtomSet(action)
  const control = useAtomSet(controlAction, { mode: "promiseExit" })

  return {
    assignResult: Result.flatMap(results, ({ assign }) => assign),
    clearResult: Result.flatMap(results, ({ clear }) => clear),
    favoriteResult: Result.flatMap(results, ({ favorite }) => favorite),
    assign: useCallback((slotId: SlotId, selection: SlotSelection) => {
      invoke({ _tag: "Assign", slotId, selection })
    }, [invoke]),
    clear: useCallback((slotId: SlotId) => {
      invoke({ _tag: "Clear", slotId })
    }, [invoke]),
    load: useCallback(
      (slotId: SlotId) => control({ _tag: "Load", slotId }),
      [control],
    ),
    stop: useCallback(
      (slotId: SlotId) => control({ _tag: "Stop", slotId }),
      [control],
    ),
    setFavorite: useCallback((model: ProviderModelIdentity, favoriteValue: boolean) => {
      invoke({ _tag: "Favorite", model, favorite: favoriteValue })
    }, [invoke]),
  }
}
