import { useCallback, useMemo } from "react"
import type { Equivalence } from "effect"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  LocalInferenceHardwareMirror,
  LocalModelsMirror,
  ModelSlotsMirror,
  ProviderModelCatalogMirror,
  type ModelInstanceId,
  type DownloadAttemptId,
  type LocalModelsState,
  type ModelServingConfigurationId,
  type ProviderModelId,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredState, useMirroredStateSelector } from "./use-mirrored-state"

export const useLocalInferenceHardware = () =>
  Result.map(useMirroredState(LocalInferenceHardwareMirror), ({ state }) => state)
export type LocalInferenceHardwareResult = ReturnType<typeof useLocalInferenceHardware>

export const useLocalModels = () =>
  Result.map(useMirroredState(LocalModelsMirror), ({ state }) => state)

export const useLocalModelsSelector = <Selection,>(
  selector: (state: LocalModelsState) => Selection,
  equivalent: Equivalence.Equivalence<Selection>,
) => {
  const snapshotSelector = useCallback(
    ({ state }: { readonly state: LocalModelsState }) => selector(state),
    [selector],
  )
  return useMirroredStateSelector(
    LocalModelsMirror,
    snapshotSelector,
    equivalent,
  )
}

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
  const installAtom = useMemo(
    () => client.mutation("InstallModel"),
    [client],
  )
  const installResult = useAtomValue(installAtom)
  const install = useAtomSet(
    installAtom,
    { mode: "promise" },
  )
  const cancel = useAtomSet(client.mutation("CancelModelDownload"))
  const dismiss = useAtomSet(client.mutation("DismissModelDownloadFailure"))
  const deleteModel = useAtomSet(client.mutation("DeleteLocalModel"))

  return {
    installResult,
    install: useCallback((configurationId: ModelServingConfigurationId) => install({
      payload: { configurationId },
      reactivityKeys: [LocalModelsMirror.id, ProviderModelCatalogMirror.id],
    }), [install]),
    cancel: useCallback((attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]]) => cancel({
      payload: { attemptIds },
      reactivityKeys: [LocalModelsMirror.id],
    }), [cancel]),
    dismissFailure: useCallback((attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]]) => dismiss({
      payload: { attemptIds },
      reactivityKeys: [LocalModelsMirror.id],
    }), [dismiss]),
    delete: useCallback((configurationId: ModelServingConfigurationId) => deleteModel({
      payload: { configurationId },
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
