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
  type ProviderId,
  type ProviderModelIdentity,
  type ProviderModelId,
  type ReasoningEffort,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { modelSlotAtoms, presentedSlotSelection } from "../model-slots/atoms"
import { useMirroredState } from "./use-mirrored-state"
import { useModelSlotsResultAtom } from "./use-local-inference-state"

export function useModelConfig() {
  const client = useAgentClient()
  const catalog = useMirroredState(ProviderModelCatalogMirror)
  const slots = useAtomValue(useModelSlotsResultAtom())
  const slotAtoms = useMemo(() => modelSlotAtoms(client), [client])
  const refreshAtom = useMemo(() => client.mutation("RefreshModelCatalog"), [client])
  const slotUpdate = useAtomValue(slotAtoms.assignMutation)
  const slotClear = useAtomValue(slotAtoms.clearMutation)
  const catalogRefresh = useAtomValue(refreshAtom)
  const favoriteUpdate = useAtomValue(slotAtoms.favoriteMutation)
  const assignmentMutationStates = useAtomValue(slotAtoms.assignmentMutationStatesAtom)
  const assign = useAtomSet(slotAtoms.assignMutation)
  const clear = useAtomSet(slotAtoms.clearMutation)
  const refresh = useAtomSet(refreshAtom)
  const setFavoriteMutation = useAtomSet(slotAtoms.favoriteMutation)
  const selections = Option.map(Result.value(slots), ({ state }) => ({
    primary: presentedSlotSelection(state, assignmentMutationStates, PRIMARY_SLOT_ID),
    secondary: presentedSlotSelection(state, assignmentMutationStates, SECONDARY_SLOT_ID),
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
    onNone: () => clear({ slotId }),
    onSome: (value) => assign({ slotId, selection: value }),
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
  ): void => setFavoriteMutation({ model, favorite }), [setFavoriteMutation])

  return {
    catalog,
    slots,
    slotUpdate,
    slotClear,
    selections,
    assignmentMutationStates,
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
