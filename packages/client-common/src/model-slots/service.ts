import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer, Option } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import {
  AcnRpcClientTag,
  ModelSlotsMirror,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type ModelSlotsState,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { ClientEffectQuery } from "../state/client-effect-query"
import { EffectQueryInvalidations } from "../state/effect-query-invalidations"

export class ModelSlotSynchronizationFailed extends Data.TaggedError(
  "ModelSlotSynchronizationFailed",
)<{
  readonly operation: "assign" | "load" | "stop"
  readonly message: string
}> {}

interface AssignmentInput {
  readonly slotId: SlotId
  readonly selection: SlotSelection
}

interface SlotInput {
  readonly slotId: SlotId
}

interface FavoriteInput {
  readonly model: ProviderModelIdentity
  readonly favorite: boolean
}

const slotSelectionScope = (slotId: SlotId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-slot-selection:${slotId}`)

const slotLoadScope = (slotId: SlotId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-slot-load:${slotId}`)

const slotStopScope = (slotId: SlotId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-slot-stop:${slotId}`)

const favoriteScope = ({ providerId, providerModelId }: ProviderModelIdentity) =>
  Mutation.MutationScope(`model-favorite:${providerId}:${providerModelId}`)

const modelSlotsQuery = Query.make("ModelSlots", {
  key: (_: void) => Data.tuple(ModelSlotsMirror.id),
  staleTime: Infinity,
  gcTime: Infinity,
  effect: () => Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("GetModelSlots", {})),
})

const synchronizeModelSlots = () => QueryClient.invalidate(modelSlotsQuery.match()).pipe(
  Effect.zipRight(QueryClient.fetch(modelSlotsQuery, undefined)),
)

export const sameSlotSelection = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId
  && left.reasoningEffort === right.reasoningEffort

export const authoritativeSlotSelection = (
  state: ModelSlotsState,
  slotId: SlotId,
): Option.Option<SlotSelection> => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag === "Unassigned" ? Option.none() : Option.some(slot.selection)
}

export const slotAssignmentIsVisible = (
  state: ModelSlotsState,
  slotId: SlotId,
  selection: SlotSelection,
): boolean => Option.exists(authoritativeSlotSelection(state, slotId), (current) =>
  sameSlotSelection(current, selection))

export const modelLoadIsVisible = (
  state: ModelSlotsState,
  slotId: SlotId,
): boolean => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag === "ConfiguredLocal"
    && slot.residency._tag !== "Unloaded"
}

export const selectedModelStopIsVisible = (
  state: ModelSlotsState,
  slotId: SlotId,
): boolean => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag !== "ConfiguredLocal"
    || slot.residency._tag !== "Requested"
      && slot.residency._tag !== "Loading"
      && slot.residency._tag !== "Ready"
}

const assignMutation = Mutation.make("AssignSlot", {
  scope: ({ slotId }: AssignmentInput) => slotSelectionScope(slotId),
  effect: ({ slotId, selection }: AssignmentInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("AssignSlot", { slotId, selection })),
  synchronize: (_, { slotId, selection }) => synchronizeModelSlots().pipe(
    Effect.filterOrFail(
      ({ state }) => slotAssignmentIsVisible(state, slotId, selection),
      () => new ModelSlotSynchronizationFailed({
        operation: "assign",
        message: "The assigned model selection was absent from ModelSlots.",
      }),
    ),
    Effect.asVoid,
  ),
})

const clearMutation = Mutation.make("ClearSlot", {
  scope: ({ slotId }: SlotInput) => slotSelectionScope(slotId),
  effect: ({ slotId }: SlotInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("ClearSlot", { slotId })),
  synchronize: () => synchronizeModelSlots().pipe(Effect.asVoid),
})

const loadMutation = Mutation.make("LoadModel", {
  scope: ({ slotId }: SlotInput) => slotLoadScope(slotId),
  effect: ({ slotId }: SlotInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("LoadModel", { slotId })),
  synchronize: (_, { slotId }) => synchronizeModelSlots().pipe(
    Effect.filterOrFail(
      ({ state }) => modelLoadIsVisible(state, slotId),
      () => new ModelSlotSynchronizationFailed({
        operation: "load",
        message: "The model load request was absent from ModelSlots.",
      }),
    ),
    Effect.asVoid,
  ),
})

const stopMutation = Mutation.make("StopModel", {
  scope: ({ slotId }: SlotInput) => slotStopScope(slotId),
  effect: ({ slotId }: SlotInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("StopModel", { slotId })),
  synchronize: (_, { slotId }) => synchronizeModelSlots().pipe(
    Effect.filterOrFail(
      ({ state }) => selectedModelStopIsVisible(state, slotId),
      () => new ModelSlotSynchronizationFailed({
        operation: "stop",
        message: "The selected model remained active in ModelSlots.",
      }),
    ),
    Effect.asVoid,
  ),
})

const favoriteMutation = Mutation.make("SetModelFavorite", {
  scope: ({ model }: FavoriteInput) => favoriteScope(model),
  effect: ({ model, favorite }: FavoriteInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("SetModelFavorite", { model, favorite })),
  synchronize: () => synchronizeModelSlots().pipe(Effect.asVoid),
})

interface PendingAssignment {
  readonly slotId: SlotId
  readonly selection: SlotSelection
  readonly pending: boolean
}

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
  const invalidations = yield* EffectQueryInvalidations
  const query = effectQuery.query(modelSlotsQuery, undefined)
  const assign = effectQuery.mutation(assignMutation)
  const clear = effectQuery.mutation(clearMutation)
  const load = effectQuery.mutation(loadMutation)
  const stop = effectQuery.mutation(stopMutation)
  const favorite = effectQuery.mutation(favoriteMutation)
  const assignResult = Atom.make((get) => get(assign))
  const clearResult = Atom.make((get) => get(clear))
  const favoriteResult = Atom.make((get) => get(favorite))
  const assignments = Mutation.state({
    filters: { mutation: assignMutation },
    select: ({ input, result }) => ({
      slotId: input.slotId,
      selection: input.selection,
      pending: Result.isWaiting(result),
    }),
  })
  const invalidate = () => queryClient.invalidate(modelSlotsQuery.match())
  yield* invalidations.register(ModelSlotsMirror.id, invalidate)
  const state = Atom.make((get) => get(query).result)
  const selections = Atom.make((get) => Result.map(get(state), ({ state: current }) => ({
    primary: presentedSlotSelection(current, get(assignments), PRIMARY_SLOT_ID),
    secondary: presentedSlotSelection(
      current,
      get(assignments),
      SECONDARY_SLOT_ID,
    ),
  })))

  return {
    state,
    selections,
    assignResult,
    clearResult,
    favoriteResult,
    retry: queryClient.invalidate(modelSlotsQuery.match()),
    assign: (slotId: SlotId, selection: SlotSelection) =>
      Mutation.execute(assign, { slotId, selection }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    clear: (slotId: SlotId) => Mutation.execute(clear, { slotId }).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
    ),
    load: (slotId: SlotId) =>
      Mutation.execute(load, { slotId }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    stop: (slotId: SlotId) => Mutation.execute(stop, { slotId }).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
    ),
    setFavorite: (model: ProviderModelIdentity, favoriteValue: boolean) =>
      Mutation.execute(favorite, { model, favorite: favoriteValue }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
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
  const service = useMemo(() => client.effectQuery.runtime.atom(ModelSlots), [client])
  const resultAtom = useMemo(() => Atom.make((get) => Result.map(get(service), (slots) => ({
    assign: get(slots.assignResult),
    clear: get(slots.clearResult),
    favorite: get(slots.favoriteResult),
  }))), [service])
  const action = useMemo(() => client.effectQuery.runtime.fn<
    | { readonly _tag: "Assign"; readonly slotId: SlotId; readonly selection: SlotSelection }
    | { readonly _tag: "Clear"; readonly slotId: SlotId }
    | { readonly _tag: "Favorite"; readonly model: ProviderModelIdentity; readonly favorite: boolean }
  >()((input) => Effect.flatMap(ModelSlots, (slots): Effect.Effect<unknown, unknown> => {
    switch (input._tag) {
      case "Assign": return slots.assign(input.slotId, input.selection)
      case "Clear": return slots.clear(input.slotId)
      case "Favorite": return slots.setFavorite(input.model, input.favorite)
    }
  })), [client])
  const controlAction = useMemo(() => client.effectQuery.runtime.fn<
    | { readonly _tag: "Load"; readonly slotId: SlotId }
    | { readonly _tag: "Stop"; readonly slotId: SlotId }
  >()((input) => Effect.flatMap(ModelSlots, (slots): Effect.Effect<unknown, unknown> =>
    input._tag === "Load" ? slots.load(input.slotId) : slots.stop(input.slotId))), [client])
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
