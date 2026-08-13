import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Data, Effect, Layer, Option } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import {
  AcnRpcClientTag,
  ModelSlotsMirror,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type ModelInstanceId,
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

interface LoadInput extends SlotInput {
  readonly selection: SlotSelection
}

interface InstanceInput {
  readonly instanceId: ModelInstanceId
}

interface FavoriteInput {
  readonly model: ProviderModelIdentity
  readonly favorite: boolean
}

const slotScope = (slotId: SlotId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-slot:${slotId}`)

const instanceScope = (instanceId: ModelInstanceId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-instance:${instanceId}`)

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

export const admittedInstanceIsVisible = (
  state: ModelSlotsState,
  slotId: SlotId,
  instanceId: ModelInstanceId,
): boolean => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag === "ConfiguredLocal"
    && Option.exists(slot.instance, (instance) => instance.id === instanceId)
}

const assignMutation = Mutation.make("AssignSlot", {
  scope: ({ slotId }: AssignmentInput) => slotScope(slotId),
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
  scope: ({ slotId }: SlotInput) => slotScope(slotId),
  effect: ({ slotId }: SlotInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("ClearSlot", { slotId })),
  synchronize: () => synchronizeModelSlots().pipe(Effect.asVoid),
})

const loadMutation = Mutation.make("LoadModel", {
  scope: ({ slotId }: LoadInput) => slotScope(slotId),
  effect: ({ slotId, selection }: LoadInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("LoadModel", { slotId, selection })),
  synchronize: ({ instanceId }, { slotId, selection }) => synchronizeModelSlots().pipe(
    Effect.filterOrFail(
      ({ state }) => slotAssignmentIsVisible(state, slotId, selection)
        && admittedInstanceIsVisible(state, slotId, instanceId),
      () => new ModelSlotSynchronizationFailed({
        operation: "load",
        message: "The admitted model instance was absent from ModelSlots.",
      }),
    ),
    Effect.asVoid,
  ),
})

const stopMutation = Mutation.make("StopModel", {
  scope: ({ instanceId }: InstanceInput) => instanceScope(instanceId),
  effect: ({ instanceId }: InstanceInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("StopModel", { instanceId })),
  synchronize: (_, { instanceId }) => synchronizeModelSlots().pipe(
    Effect.filterOrFail(
      ({ state }) => [state.slots.primary, state.slots.secondary].every((slot) =>
        slot._tag !== "ConfiguredLocal"
        || Option.isNone(slot.instance)
        || slot.instance.value.id !== instanceId
        || slot.instance.value.lifecycle._tag === "Stopped"
        || slot.instance.value.lifecycle._tag === "Failed"),
      () => new ModelSlotSynchronizationFailed({
        operation: "stop",
        message: "The stopped model instance remained active in ModelSlots.",
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
    assign: (slotId: SlotId, selection: SlotSelection) =>
      Mutation.execute(assign, { slotId, selection }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    clear: (slotId: SlotId) => Mutation.execute(clear, { slotId }).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
    ),
    load: (slotId: SlotId, selection: SlotSelection) =>
      Mutation.execute(load, { slotId, selection }).pipe(
        Effect.provideService(Registry.AtomRegistry, registry),
      ),
    stop: (instanceId: ModelInstanceId) => Mutation.execute(stop, { instanceId }).pipe(
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
    | { readonly _tag: "Load"; readonly slotId: SlotId; readonly selection: SlotSelection }
    | { readonly _tag: "Stop"; readonly instanceId: ModelInstanceId }
    | { readonly _tag: "Favorite"; readonly model: ProviderModelIdentity; readonly favorite: boolean }
  >()((input) => Effect.flatMap(ModelSlots, (slots): Effect.Effect<unknown, unknown> => {
    switch (input._tag) {
      case "Assign": return slots.assign(input.slotId, input.selection)
      case "Clear": return slots.clear(input.slotId)
      case "Load": return slots.load(input.slotId, input.selection)
      case "Stop": return slots.stop(input.instanceId)
      case "Favorite": return slots.setFavorite(input.model, input.favorite)
    }
  })), [client])
  const results = useAtomValue(resultAtom)
  const invoke = useAtomSet(action)

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
    load: useCallback((slotId: SlotId, selection: SlotSelection) => {
      invoke({ _tag: "Load", slotId, selection })
    }, [invoke]),
    stop: useCallback((instanceId: ModelInstanceId) => {
      invoke({ _tag: "Stop", instanceId })
    }, [invoke]),
    setFavorite: useCallback((model: ProviderModelIdentity, favoriteValue: boolean) => {
      invoke({ _tag: "Favorite", model, favorite: favoriteValue })
    }, [invoke]),
  }
}
