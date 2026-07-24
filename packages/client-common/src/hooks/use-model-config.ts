import { useMemo } from "react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
  ModelSlotsMirror,
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  ProviderModelCatalogMirror,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type ModelSlotsState,
  type ProviderId,
  type ProviderModelIdentity,
  type ProviderModelId,
  type ReasoningEffort,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredState } from "./use-mirrored-state"

const selectionAt = (state: ModelSlotsState, slotId: SlotId): Option.Option<SlotSelection> => {
  const slot = state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]
  return slot._tag === "Unassigned" ? Option.none() : Option.some(slot.selection)
}

export function useModelConfig() {
  const client = useAgentClient()
  const catalog = useMirroredState(ProviderModelCatalogMirror)
  const slots = useMirroredState(ModelSlotsMirror)
  const assignAtom = useMemo(() => client.mutation("AssignSlot"), [client])
  const clearAtom = useMemo(() => client.mutation("ClearSlot"), [client])
  const refreshAtom = useMemo(() => client.mutation("RefreshModelCatalog"), [client])
  const favoriteAtom = useMemo(() => client.mutation("SetModelFavorite"), [client])
  const slotUpdate = useAtomValue(assignAtom)
  const slotClear = useAtomValue(clearAtom)
  const catalogRefresh = useAtomValue(refreshAtom)
  const favoriteUpdate = useAtomValue(favoriteAtom)
  const assign = useAtomSet(assignAtom)
  const clear = useAtomSet(clearAtom)
  const refresh = useAtomSet(refreshAtom)
  const setFavoriteMutation = useAtomSet(favoriteAtom)

  const selections = Option.map(Result.value(slots), ({ state }) => ({
    primary: selectionAt(state, PRIMARY_SLOT_ID),
    secondary: selectionAt(state, SECONDARY_SLOT_ID),
  }))

  const catalogModels = Option.flatMap(Result.value(catalog), ({ state }) =>
    ProviderModelCatalogLifecycle.match(state, {
      Loading: () => Option.none(),
      Ready: ({ models }) => Option.some(models),
      Refreshing: ({ models }) => Option.some(models),
      Degraded: ({ models }) => Option.some(models),
      Unavailable: () => Option.none(),
    }))

  const commit = useMemo(() => (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
  ): void => Option.match(selection, {
    onNone: () => clear({
      payload: { slotId },
      reactivityKeys: [ModelSlotsMirror.id],
    }),
    onSome: (value) => assign({
      payload: { slotId, selection: value },
      reactivityKeys: [ModelSlotsMirror.id],
    }),
  }), [assign, clear])

  const selectionFor = useMemo(() => (
    slotId: SlotId,
    providerId: ProviderId,
    providerModelId: ProviderModelId,
  ): SlotSelection => {
    const current = Option.flatMap(selections, (values) => slotId === PRIMARY_SLOT_ID ? values.primary : values.secondary)
    const model = Option.flatMap(catalogModels, (models) => Option.fromNullable(models.find((candidate) =>
      candidate.providerId === providerId && candidate.providerModelId === providerModelId)))
    const currentEffort = Option.filter(current, (value) => value.providerId === providerId
      && value.providerModelId === providerModelId)
    const reasoningEffort = Option.match(model, {
      onNone: () => Option.match(currentEffort, {
        onSome: (value) => value.reasoningEffort,
        onNone: () => ReasoningEffortSchema.make("none"),
      }),
      onSome: (value) => Option.match(currentEffort, {
        onSome: (currentSelection) =>
          value.capabilities.reasoning.efforts.includes(currentSelection.reasoningEffort)
            ? currentSelection.reasoningEffort
            : Option.getOrElse(
                value.capabilities.reasoning.defaultEffort,
                () => ReasoningEffortSchema.make("none"),
              ),
        onNone: () => Option.getOrElse(
          value.capabilities.reasoning.defaultEffort,
          () => ReasoningEffortSchema.make("none"),
        ),
      }),
    })
    return {
      providerId,
      providerModelId,
      reasoningEffort,
    }
  }, [catalogModels, selections])

  const updateSlotModel = useMemo(() => (
    slotId: SlotId,
    providerId: ProviderId,
    providerModelId: ProviderModelId,
  ): void => commit(slotId, Option.some(selectionFor(slotId, providerId, providerModelId))), [commit, selectionFor])

  const clearSlot = useMemo(() => (slotId: SlotId) => commit(slotId, Option.none()), [commit])

  const updateSlotReasoning = useMemo(() => (slotId: SlotId, effort: ReasoningEffort): void => {
    const current = Option.flatMap(selections, (values) => slotId === PRIMARY_SLOT_ID ? values.primary : values.secondary)
    if (Option.isNone(current)) return
    commit(slotId, Option.some({ ...current.value, reasoningEffort: effort }))
  }, [commit, selections])

  const favoriteModels = Option.match(Result.value(slots), {
    onNone: () => [] as readonly ProviderModelIdentity[],
    onSome: ({ state }) => state.favoriteModels,
  })
  const setModelFavorite = useMemo(() => (
    model: ProviderModelIdentity,
    favorite: boolean,
  ): void => setFavoriteMutation({
    payload: { model, favorite },
    reactivityKeys: [ModelSlotsMirror.id],
  }), [setFavoriteMutation])

  return {
    catalog,
    slots,
    slotUpdate,
    slotClear,
    catalogRefresh,
    favoriteUpdate,
    favoriteModels,
    setModelFavorite,
    updateSlotModel,
    clearSlot,
    updateSlotReasoning,
    resetToDefaults: () => {
      clearSlot(PRIMARY_SLOT_ID)
      clearSlot(SECONDARY_SLOT_ID)
    },
    refreshModels: () => refresh({
      payload: { providerId: Option.none() },
      reactivityKeys: [ProviderModelCatalogMirror.id, ModelSlotsMirror.id],
    }),
  }
}

export type UseModelConfigResult = ReturnType<typeof useModelConfig>
