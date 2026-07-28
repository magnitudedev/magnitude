import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue, type Atom as EffectAtom } from "@effect-atom/atom-react"
import {
  LocalInferenceHardwareMirror,
  LocalModelsMirror,
  ModelSlotsMirror,
  ProviderModelCatalogMirror,
  type ModelInstanceId,
  type ModelOfferingTargetId,
  type ProviderModelId,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
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
  const downloadAtom = useMemo(() => client.mutation("DownloadModel"), [client])
  const cancelAtom = useMemo(() => client.mutation("CancelModelDownload"), [client])
  const dismissAtom = useMemo(() => client.mutation("DismissModelDownloadFailure"), [client])
  const deleteAtom = useMemo(() => client.mutation("DeleteLocalModel"), [client])
  const assignAtom = useMemo(() => client.mutation("AssignSlot"), [client])
  const clearAtom = useMemo(() => client.mutation("ClearSlot"), [client])
  const loadAtom = useMemo(() => client.mutation("LoadModel"), [client])
  const stopAtom = useMemo(() => client.mutation("StopModel"), [client])
  const downloadResult = useAtomValue(downloadAtom)
  const cancelDownloadResult = useAtomValue(cancelAtom)
  const dismissDownloadFailureResult = useAtomValue(dismissAtom)
  const deleteLocalModelResult = useAtomValue(deleteAtom)
  const slotAssignment = useAtomValue(assignAtom)
  const clearSlotResult = useAtomValue(clearAtom)
  const loadModelResult = useAtomValue(loadAtom)
  const stopModelResult = useAtomValue(stopAtom)
  const download = useAtomSet(downloadAtom, { mode: "promise" })
  const cancel = useAtomSet(cancelAtom)
  const dismiss = useAtomSet(dismissAtom)
  const deleteModel = useAtomSet(deleteAtom)
  const assign = useAtomSet(assignAtom, { mode: "promise" })
  const clear = useAtomSet(clearAtom)
  const load = useAtomSet(loadAtom, { mode: "promise" })
  const stop = useAtomSet(stopAtom)
  const modelKeys = [LocalModelsMirror.id, ProviderModelCatalogMirror.id] as const
  return {
    state,
    downloadResult,
    cancelDownloadResult,
    dismissDownloadFailureResult,
    deleteLocalModelResult,
    slotAssignment,
    clearSlotResult,
    loadModelResult,
    stopModelResult,
    downloadModel: useCallback((targetId: ModelOfferingTargetId) =>
      download({
        payload: { targetId },
        reactivityKeys: modelKeys,
      }), [download]),
    cancelModelDownload: useCallback((targetId: ModelOfferingTargetId) => cancel({
      payload: { targetId },
      reactivityKeys: [LocalModelsMirror.id],
    }), [cancel]),
    dismissModelDownloadFailure: useCallback((targetId: ModelOfferingTargetId) => dismiss({
      payload: { targetId },
      reactivityKeys: [LocalModelsMirror.id],
    }), [dismiss]),
    deleteLocalModel: useCallback((targetId: ModelOfferingTargetId) => deleteModel({
      payload: { targetId },
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
    stopModel: useCallback((instanceId: ModelInstanceId) => stop({
      payload: { instanceId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [stop]),
  }
}
