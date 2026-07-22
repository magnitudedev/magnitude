import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue, type Atom as EffectAtom } from "@effect-atom/atom-react"
import {
  LocalInferenceHardwareMirror,
  LocalModelInventoryMirror,
  ModelSlotsMirror,
  type LocalModelId,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { Option } from "effect"
import { useAgentClient } from "../state/agent-client-context"
import type { LocalInferenceView } from "../types/local-inference"
import { useMirroredState, useMirroredStateAtom } from "./use-mirrored-state"

type HardwareSnapshot = typeof LocalInferenceHardwareMirror.snapshotSchema.Type
type InventorySnapshot = typeof LocalModelInventoryMirror.snapshotSchema.Type
type SlotsSnapshot = typeof ModelSlotsMirror.snapshotSchema.Type

export const makeLocalInferenceQueryAtom = <E>(
  hardware: EffectAtom.Atom<Result.Result<HardwareSnapshot, E>>,
  inventory: EffectAtom.Atom<Result.Result<InventorySnapshot, E>>,
  slots: EffectAtom.Atom<Result.Result<SlotsSnapshot, E>>,
) => Atom.make((get) => Result.map(Result.all({
  hardware: get(hardware),
  inventory: get(inventory),
  slots: get(slots),
}), ({ hardware, inventory, slots }): LocalInferenceView => ({
  hardware: hardware.state,
  inventory: inventory.state,
  slots: slots.state,
})))

export const useLocalInferenceHardware = () => useMirroredState(LocalInferenceHardwareMirror)
export const useLocalModelInventory = () => useMirroredState(LocalModelInventoryMirror)
export const useModelSlots = () => useMirroredState(ModelSlotsMirror)

export function useLocalInferenceQuery() {
  const hardware = useMirroredStateAtom(LocalInferenceHardwareMirror)
  const inventory = useMirroredStateAtom(LocalModelInventoryMirror)
  const slots = useMirroredStateAtom(ModelSlotsMirror)
  const state = useMemo(
    () => makeLocalInferenceQueryAtom(hardware, inventory, slots),
    [hardware, inventory, slots],
  )
  return useAtomValue(state)
}

export function useLocalInferenceState() {
  const client = useAgentClient()
  const state = useLocalInferenceQuery()
  const downloadAtom = useMemo(() => client.mutation("DownloadLocalModel"), [client])
  const deleteAtom = useMemo(() => client.mutation("DeleteLocalModel"), [client])
  const loadAtom = useMemo(() => client.mutation("LoadModelSlot"), [client])
  const unloadAtom = useMemo(() => client.mutation("UnloadModelSlot"), [client])
  const reloadAtom = useMemo(() => client.mutation("ReloadModelSlot"), [client])
  const downloadResult = useAtomValue(downloadAtom)
  const deleteResult = useAtomValue(deleteAtom)
  const loadResult = useAtomValue(loadAtom)
  const unloadResult = useAtomValue(unloadAtom)
  const reloadResult = useAtomValue(reloadAtom)
  const download = useAtomSet(downloadAtom)
  const remove = useAtomSet(deleteAtom)
  const load = useAtomSet(loadAtom)
  const unload = useAtomSet(unloadAtom)
  const reload = useAtomSet(reloadAtom)
  return {
    state,
    mutationFailure: Option.fromNullable(
      [downloadResult, deleteResult, loadResult, unloadResult, reloadResult].find(Result.isFailure),
    ),
    downloadModel: useCallback((localModelId: LocalModelId) => download({
      payload: { localModelId },
      reactivityKeys: [LocalModelInventoryMirror.id],
    }), [download]),
    deleteModel: useCallback((localModelId: LocalModelId) => remove({
      payload: { localModelId },
      reactivityKeys: [LocalModelInventoryMirror.id, ModelSlotsMirror.id],
    }), [remove]),
    loadSlot: useCallback((slotId: SlotId, selection: SlotSelection) => load({
      payload: { slotId, selection },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [load]),
    unloadSlot: useCallback((slotId: SlotId) => unload({
      payload: { slotId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [unload]),
    reloadSlot: useCallback((slotId: SlotId) => reload({
      payload: { slotId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [reload]),
  }
}
