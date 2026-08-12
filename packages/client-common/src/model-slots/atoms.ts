import { Atom, Result } from "@effect-atom/atom-react"
import { Data, Effect, Option } from "effect"
import { Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import {
  AcnRpcClientTag,
  ModelSlotsMirror,
  PRIMARY_SLOT_ID,
  type ModelInstanceId,
  type ModelSlotsState,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import type { AgentClientInstance } from "../state/agent-client"
import {
  getMirroredStateInvalidationWatch,
  subscribeToMirroredStateInvalidation,
} from "../hooks/use-mirrored-state"

export interface ModelSlotAssignmentInput {
  readonly slotId: SlotId
  readonly selection: SlotSelection
}

export interface ModelSlotInput {
  readonly slotId: SlotId
}

export interface ModelInstanceInput {
  readonly instanceId: ModelInstanceId
}

export interface ModelFavoriteInput {
  readonly model: ProviderModelIdentity
  readonly favorite: boolean
}

export const modelSlotMutationScope = (slotId: SlotId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-slot:${slotId}`)

const modelInstanceMutationScope = (instanceId: ModelInstanceId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-instance:${instanceId}`)

const modelFavoriteMutationScope = ({ providerId, providerModelId }: ProviderModelIdentity) =>
  Mutation.MutationScope(`model-favorite:${providerId}:${providerModelId}`)

export const modelSlotsQuery = Query.make("ModelSlots", {
  key: (_: void) => Data.tuple("model-slots"),
  staleTime: Infinity,
  gcTime: Infinity,
  effect: () => Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("GetModelSlots", {})),
})

const synchronizeModelSlots = () => QueryClient.invalidate(
  modelSlotsQuery.match(),
).pipe(
  Effect.zipRight(QueryClient.fetch(modelSlotsQuery, undefined)),
  Effect.asVoid,
)

export const assignModelSlotMutation = Mutation.make("AssignSlot", {
  scope: ({ slotId }: ModelSlotAssignmentInput) => modelSlotMutationScope(slotId),
  effect: ({ slotId, selection }: ModelSlotAssignmentInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("AssignSlot", { slotId, selection })),
  synchronize: synchronizeModelSlots,
})

export const clearModelSlotMutation = Mutation.make("ClearSlot", {
  scope: ({ slotId }: ModelSlotInput) => modelSlotMutationScope(slotId),
  effect: ({ slotId }: ModelSlotInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("ClearSlot", { slotId })),
  synchronize: synchronizeModelSlots,
})

export const loadModelMutation = Mutation.make("LoadModel", {
  scope: ({ slotId }: ModelSlotInput) => modelSlotMutationScope(slotId),
  effect: ({ slotId }: ModelSlotInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("LoadModel", { slotId })),
  synchronize: synchronizeModelSlots,
})

export const stopModelMutation = Mutation.make("StopModel", {
  scope: ({ instanceId }: ModelInstanceInput) => modelInstanceMutationScope(instanceId),
  effect: ({ instanceId }: ModelInstanceInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("StopModel", { instanceId })),
  synchronize: synchronizeModelSlots,
})

export const setModelFavoriteMutation = Mutation.make("SetModelFavorite", {
  scope: ({ model }: ModelFavoriteInput) => modelFavoriteMutationScope(model),
  effect: ({ model, favorite }: ModelFavoriteInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("SetModelFavorite", { model, favorite })),
  synchronize: synchronizeModelSlots,
})

const makeAtoms = (client: AgentClientInstance) => {
  const effectQuery = client.effectQuery
  const modelSlotsQueryAtom = effectQuery.query(modelSlotsQuery, undefined)
  const assignMutation = effectQuery.mutation(assignModelSlotMutation)

  const invalidationBridgeEffect = Effect.gen(function* () {
    const queryClient = yield* QueryClient.QueryClient
    yield* Effect.acquireRelease(
      Effect.sync(() => subscribeToMirroredStateInvalidation(
        client,
        ModelSlotsMirror.id,
        () => queryClient.invalidate(modelSlotsQuery.match()),
      )),
      (unsubscribe) => Effect.sync(unsubscribe),
    )
    yield* queryClient.prefetch(modelSlotsQueryAtom)
    return yield* Effect.never
  })

  return {
    modelSlotsQueryAtom,
    modelSlotsResultAtom: Atom.make((get) => get(modelSlotsQueryAtom).result),
    assignMutation,
    assignmentMutationStatesAtom: Mutation.state({
      filters: { mutation: assignModelSlotMutation },
    }),
    clearMutation: effectQuery.mutation(clearModelSlotMutation),
    loadMutation: effectQuery.mutation(loadModelMutation),
    stopMutation: effectQuery.mutation(stopModelMutation),
    favoriteMutation: effectQuery.mutation(setModelFavoriteMutation),
    invalidationBridgeAtom: effectQuery.runtime.atom(invalidationBridgeEffect),
    mirrorInvalidationWatchAtom: getMirroredStateInvalidationWatch(client, ModelSlotsMirror.id),
  }
}

export type ModelSlotAtoms = ReturnType<typeof makeAtoms>
export type ModelSlotAssignmentMutationState = Mutation.State<typeof assignModelSlotMutation>

export const latestPendingModelSlotAssignment = (
  mutationStates: ReadonlyArray<ModelSlotAssignmentMutationState>,
  slotId: SlotId,
): ModelSlotAssignmentMutationState | undefined => mutationStates.findLast(({ input, result }) =>
  input.slotId === slotId && Result.isWaiting(result))

export const authoritativeSlotSelection = (
  state: ModelSlotsState,
  slotId: SlotId,
): Option.Option<SlotSelection> => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag === "Unassigned" ? Option.none() : Option.some(slot.selection)
}

export const presentedSlotSelection = (
  state: ModelSlotsState,
  mutationStates: ReadonlyArray<ModelSlotAssignmentMutationState>,
  slotId: SlotId,
): Option.Option<SlotSelection> => {
  const pending = latestPendingModelSlotAssignment(mutationStates, slotId)
  return pending === undefined ? authoritativeSlotSelection(state, slotId) : Option.some(pending.input.selection)
}

const atomsByClient = new WeakMap<object, ModelSlotAtoms>()

export const modelSlotAtoms = (client: AgentClientInstance): ModelSlotAtoms => {
  const existing = atomsByClient.get(client)
  if (existing !== undefined) return existing
  const atoms = makeAtoms(client)
  atomsByClient.set(client, atoms)
  return atoms
}
