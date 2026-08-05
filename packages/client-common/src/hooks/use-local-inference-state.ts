import { useCallback, useMemo } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  LocalInferenceHardwareMirror,
  LocalModelsMirror,
  ModelSlotsMirror,
  ProviderModelCatalogMirror,
  type ModelInstanceId,
  type DownloadAttemptId,
  type ModelOfferingTargetId,
  type ModelServingConfigurationId,
  type ProviderModelId,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredState } from "./use-mirrored-state"

export const useLocalInferenceHardware = () =>
  Result.map(useMirroredState(LocalInferenceHardwareMirror), ({ state }) => state)
export type LocalInferenceHardwareResult = ReturnType<typeof useLocalInferenceHardware>

export const useLocalModels = () =>
  Result.map(useMirroredState(LocalModelsMirror), ({ state }) => state)

export const useModelSlots = () =>
  Result.map(useMirroredState(ModelSlotsMirror), ({ state }) => state)

export const useProviderModelCatalog = () =>
  Result.map(useMirroredState(ProviderModelCatalogMirror), ({ state }) => state)

export function usePreviewModelLoad(slotId: SlotId) {
  const client = useAgentClient()
  const preview = useMemo(
    () => client.query(
      "PreviewModelLoad",
      { slotId },
      { reactivityKeys: [LocalInferenceHardwareMirror.id, ModelSlotsMirror.id] },
    ),
    [client, slotId],
  )
  return useAtomValue(preview)
}

export function useLocalModelActions() {
  const client = useAgentClient()
  const createOfferingAtom = useMemo(
    () => client.mutation("CreateLocalModelOffering"),
    [client],
  )
  const createOfferingResult = useAtomValue(createOfferingAtom)
  const createOffering = useAtomSet(
    createOfferingAtom,
    { mode: "promise" },
  )
  const download = useAtomSet(client.mutation("DownloadModel"))
  const cancel = useAtomSet(client.mutation("CancelModelDownload"))
  const dismiss = useAtomSet(client.mutation("DismissModelDownloadFailure"))
  const deleteModel = useAtomSet(client.mutation("DeleteLocalModel"))

  return {
    createOfferingResult,
    createOffering: useCallback((configurationId: ModelServingConfigurationId) => createOffering({
      payload: { configurationId },
      reactivityKeys: [LocalModelsMirror.id, ProviderModelCatalogMirror.id],
    }), [createOffering]),
    download: useCallback((targetId: ModelOfferingTargetId) =>
      download({
        payload: { targetId },
        reactivityKeys: [LocalModelsMirror.id],
      }), [download]),
    cancel: useCallback((attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]]) => cancel({
      payload: { attemptIds },
      reactivityKeys: [LocalModelsMirror.id],
    }), [cancel]),
    dismissFailure: useCallback((targetId: ModelOfferingTargetId) => dismiss({
      payload: { targetId },
      reactivityKeys: [LocalModelsMirror.id],
    }), [dismiss]),
    delete: useCallback((targetId: ModelOfferingTargetId) => deleteModel({
      payload: { targetId },
      reactivityKeys: [
        LocalModelsMirror.id,
        ProviderModelCatalogMirror.id,
        ModelSlotsMirror.id,
      ],
    }), [deleteModel]),
  }
}

export function useModelSlotActions() {
  const client = useAgentClient()
  const assign = useAtomSet(client.mutation("AssignSlot"))
  const clear = useAtomSet(client.mutation("ClearSlot"))
  const load = useAtomSet(client.mutation("LoadModel"))
  const stop = useAtomSet(client.mutation("StopModel"))
  const favorite = useAtomSet(client.mutation("SetModelFavorite"))

  return {
    assign: useCallback((slotId: SlotId, selection: SlotSelection) => assign({
      payload: { slotId, selection },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [assign]),
    clear: useCallback((slotId: SlotId) => clear({
      payload: { slotId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [clear]),
    load: useCallback((slotId: SlotId) => load({
      payload: { slotId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [load]),
    stop: useCallback((instanceId: ModelInstanceId) => stop({
      payload: { instanceId },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [stop]),
    setFavorite: useCallback((model: ProviderModelIdentity, isFavorite: boolean) => favorite({
      payload: { model, favorite: isFavorite },
      reactivityKeys: [ModelSlotsMirror.id],
    }), [favorite]),
  }
}
