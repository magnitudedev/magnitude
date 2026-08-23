import { useCallback, useMemo } from "react"
import { Atom, Registry, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import {
  Configuration,
  LocalInference,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  authoritativeSlotSelection,
  type ModelSlotsState,
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
  const query = effectQuery.query(Configuration.GetModelSlots, {})
  const assign = effectQuery.mutation(Configuration.AssignSlot)
  const clear = effectQuery.mutation(Configuration.ClearSlot)
  const load = effectQuery.mutation(LocalInference.LoadModel)
  const stop = effectQuery.mutation(LocalInference.StopModel)
  const favorite = effectQuery.mutation(Configuration.SetModelFavorite)
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
  const state = Atom.make((get) => get(query).result)
  const selections = Atom.make((get) => Result.map(get(state), ({ state: current }) => ({
    primary: presentedSlotSelection(current, get(assignments), PRIMARY_SLOT_ID),
    secondary: presentedSlotSelection(
      current,
      get(assignments),
      SECONDARY_SLOT_ID,
    ),
  })))
  const provideRegistry = Effect.provideService(Registry.AtomRegistry, registry)

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
    load: (slotId: SlotId) => Mutation.execute(load, { slotId }).pipe(provideRegistry),
    stop: (slotId: SlotId) => Mutation.execute(stop, { slotId }).pipe(provideRegistry),
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
  >()((input) => Effect.flatMap(ModelSlots, (slots): Effect.Effect<unknown, unknown> => {
    switch (input._tag) {
      case "Assign": return slots.assign(input.slotId, input.selection)
      case "Clear": return slots.clear(input.slotId)
      case "Favorite": return slots.setFavorite(input.model, input.favorite)
    }
  })), [client])
  const controlAction = useMemo(() => client.runtime.fn<
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
