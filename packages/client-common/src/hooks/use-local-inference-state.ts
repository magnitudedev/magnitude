import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue, type Atom as EffectAtom } from "@effect-atom/atom-react"
import {
  LocalInferenceHardwareMirror,
  LocalModelsMirror,
  ModelSlotsMirror,
  ProviderModelCatalogMirror,
  type ModelOfferingTargetId,
  type RecommendationId,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { Option } from "effect"
import { useAgentClient } from "../state/agent-client-context"
import type { LocalInferenceView } from "../types/local-inference"
import { useMirroredState, useMirroredStateAtom } from "./use-mirrored-state"

type HardwareSnapshot = typeof LocalInferenceHardwareMirror.snapshotSchema.Type
type ModelsSnapshot = typeof LocalModelsMirror.snapshotSchema.Type
type CatalogSnapshot = typeof ProviderModelCatalogMirror.snapshotSchema.Type
type SlotsSnapshot = typeof ModelSlotsMirror.snapshotSchema.Type

export const makeLocalInferenceQueryAtom = <E>(
  hardware: EffectAtom.Atom<Result.Result<HardwareSnapshot, E>>,
  models: EffectAtom.Atom<Result.Result<ModelsSnapshot, E>>,
  catalog: EffectAtom.Atom<Result.Result<CatalogSnapshot, E>>,
  slots: EffectAtom.Atom<Result.Result<SlotsSnapshot, E>>,
) => Atom.make((get) => Result.map(Result.all({
  hardware: get(hardware),
  models: get(models),
  catalog: get(catalog),
  slots: get(slots),
}), ({ hardware, models, catalog, slots }): LocalInferenceView => ({
  hardware: hardware.state,
  models: models.state,
  catalog: catalog.state,
  slots: slots.state,
})))

export const useLocalInferenceHardware = () => useMirroredState(LocalInferenceHardwareMirror)
export const useLocalModels = () => useMirroredState(LocalModelsMirror)
export const useModelSlots = () => useMirroredState(ModelSlotsMirror)

export function useLocalInferenceQuery() {
  const hardware = useMirroredStateAtom(LocalInferenceHardwareMirror)
  const models = useMirroredStateAtom(LocalModelsMirror)
  const catalog = useMirroredStateAtom(ProviderModelCatalogMirror)
  const slots = useMirroredStateAtom(ModelSlotsMirror)
  const state = useMemo(
    () => makeLocalInferenceQueryAtom(hardware, models, catalog, slots),
    [hardware, models, catalog, slots],
  )
  return useAtomValue(state)
}

export function useLocalInferenceState() {
  const client = useAgentClient()
  const state = useLocalInferenceQuery()
  const downloadAtom = useMemo(() => client.mutation("DownloadRecommendedModel"), [client])
  const retryAtom = useMemo(() => client.mutation("RetryModelDownload"), [client])
  const cancelAtom = useMemo(() => client.mutation("CancelModelDownload"), [client])
  const dismissAtom = useMemo(() => client.mutation("DismissModelDownloadFailure"), [client])
  const deleteAtom = useMemo(() => client.mutation("DeleteLocalModel"), [client])
  const assignAtom = useMemo(() => client.mutation("AssignSlot"), [client])
  const clearAtom = useMemo(() => client.mutation("ClearSlot"), [client])
  const loadAtom = useMemo(() => client.mutation("LoadModel"), [client])
  const unloadAtom = useMemo(() => client.mutation("UnloadModel"), [client])
  const mutations = [
    useAtomValue(downloadAtom),
    useAtomValue(retryAtom),
    useAtomValue(cancelAtom),
    useAtomValue(dismissAtom),
    useAtomValue(deleteAtom),
    useAtomValue(assignAtom),
    useAtomValue(clearAtom),
    useAtomValue(loadAtom),
    useAtomValue(unloadAtom),
  ]
  const download = useAtomSet(downloadAtom)
  const retry = useAtomSet(retryAtom)
  const cancel = useAtomSet(cancelAtom)
  const dismiss = useAtomSet(dismissAtom)
  const deleteModel = useAtomSet(deleteAtom)
  const assign = useAtomSet(assignAtom, { mode: "promise" })
  const clear = useAtomSet(clearAtom)
  const load = useAtomSet(loadAtom)
  const unload = useAtomSet(unloadAtom)
  const modelKeys = [LocalModelsMirror.id, ProviderModelCatalogMirror.id] as const
  return {
    state,
    mutationFailure: Option.fromNullable(mutations.find(Result.isFailure)),
    downloadRecommendedModel: useCallback((recommendationId: RecommendationId) => download({
      payload: { recommendationId },
      reactivityKeys: modelKeys,
    }), [download]),
    retryModelDownload: useCallback((modelId: ModelOfferingTargetId) => retry({
      payload: { modelId },
      reactivityKeys: modelKeys,
    }), [retry]),
    cancelModelDownload: useCallback((modelId: ModelOfferingTargetId) => cancel({
      payload: { modelId },
      reactivityKeys: [LocalModelsMirror.id],
    }), [cancel]),
    dismissModelDownloadFailure: useCallback((modelId: ModelOfferingTargetId) => dismiss({
      payload: { modelId },
      reactivityKeys: [LocalModelsMirror.id],
    }), [dismiss]),
    deleteLocalModel: useCallback((modelId: ModelOfferingTargetId) => deleteModel({
      payload: { modelId },
      reactivityKeys: [LocalModelsMirror.id, ProviderModelCatalogMirror.id, ModelSlotsMirror.id],
    }), [deleteModel]),
    assignSlot: useCallback((slotId: SlotId, selection: SlotSelection) => assign({
      payload: { slotId, selection },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [assign]),
    clearSlot: useCallback((slotId: SlotId) => clear({
      payload: { slotId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [clear]),
    loadModel: useCallback((slotId: SlotId) => load({
      payload: { slotId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [load]),
    unloadModel: useCallback((slotId: SlotId) => unload({
      payload: { slotId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [unload]),
  }
}
