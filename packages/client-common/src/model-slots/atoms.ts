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

export class ModelSlotSynchronizationFailed extends Data.TaggedError(
  "ModelSlotSynchronizationFailed",
)<{
  readonly operation: "assign" | "load" | "stop"
  readonly message: string
}> {}

export interface ModelSlotAssignmentInput {
  readonly slotId: SlotId
  readonly selection: SlotSelection
}

export interface ModelSlotInput {
  readonly slotId: SlotId
}

export interface ModelSlotLoadInput extends ModelSlotInput {
  readonly selection: SlotSelection
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
)

export const sameSlotSelection = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId
  && left.reasoningEffort === right.reasoningEffort

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

export const assignModelSlotMutation = Mutation.make("AssignSlot", {
  scope: ({ slotId }: ModelSlotAssignmentInput) => modelSlotMutationScope(slotId),
  effect: ({ slotId, selection }: ModelSlotAssignmentInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("AssignSlot", { slotId, selection })),
  synchronize: (_, { slotId, selection }) => synchronizeModelSlots().pipe(
    Effect.filterOrFail(
      (state) => slotAssignmentIsVisible(state.state, slotId, selection),
      () => new ModelSlotSynchronizationFailed({
        operation: "assign",
        message: "The assigned model selection was absent from ModelSlots.",
      }),
    ),
    Effect.asVoid,
  ),
})

export const clearModelSlotMutation = Mutation.make("ClearSlot", {
  scope: ({ slotId }: ModelSlotInput) => modelSlotMutationScope(slotId),
  effect: ({ slotId }: ModelSlotInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("ClearSlot", { slotId })),
  synchronize: () => synchronizeModelSlots().pipe(Effect.asVoid),
})

export const loadModelMutation = Mutation.make("LoadModel", {
  scope: ({ slotId }: ModelSlotLoadInput) => modelSlotMutationScope(slotId),
  effect: ({ slotId, selection }: ModelSlotLoadInput) =>
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

export const stopModelMutation = Mutation.make("StopModel", {
  scope: ({ instanceId }: ModelInstanceInput) => modelInstanceMutationScope(instanceId),
  effect: ({ instanceId }: ModelInstanceInput) =>
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

export const setModelFavoriteMutation = Mutation.make("SetModelFavorite", {
  scope: ({ model }: ModelFavoriteInput) => modelFavoriteMutationScope(model),
  effect: ({ model, favorite }: ModelFavoriteInput) =>
    Effect.flatMap(AcnRpcClientTag, (rpc) => rpc("SetModelFavorite", { model, favorite })),
  synchronize: () => synchronizeModelSlots().pipe(Effect.asVoid),
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
