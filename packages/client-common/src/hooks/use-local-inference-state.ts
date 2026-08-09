import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option } from "effect"
import {
  LocalInferenceHardwareMirror,
  LocalModelsMirror,
  ModelSlotsMirror,
  ProviderIdSchema,
  ProviderModelCatalogMirror,
  type ModelInstanceId,
  type DownloadAttemptId,
  type ModelOfferingTargetId,
  type ModelServingConfigurationId,
  type ProviderModelId,
  type ProviderModelIdentity,
  type ReasoningEffort,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredState } from "./use-mirrored-state"
import { useMirroredStateAtom } from "./use-mirrored-state"

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
  const downloadAtom = useMemo(() => client.mutation("DownloadModel"), [client])
  const cancelAtom = useMemo(() => client.mutation("CancelModelDownload"), [client])
  const dismissAtom = useMemo(() => client.mutation("DismissModelDownloadFailure"), [client])
  const deleteAtom = useMemo(() => client.mutation("DeleteLocalModel"), [client])
  const downloadResult = useAtomValue(downloadAtom)
  const cancelResult = useAtomValue(cancelAtom)
  const dismissFailureResult = useAtomValue(dismissAtom)
  const deleteResult = useAtomValue(deleteAtom)
  const download = useAtomSet(downloadAtom)
  const cancel = useAtomSet(cancelAtom)
  const dismiss = useAtomSet(dismissAtom)
  const deleteModel = useAtomSet(deleteAtom)

  return {
    createOfferingResult,
    downloadResult,
    cancelResult,
    dismissFailureResult,
    deleteResult,
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

export interface LocalConfigurationSelection {
  readonly slotId: SlotId
  readonly targetId: ModelOfferingTargetId
  readonly configurationId: ModelServingConfigurationId
  readonly reasoningEffort: ReasoningEffort
}

export const findLocalConfigurationOffering = (
  models: { readonly models: readonly {
    readonly targetId: ModelOfferingTargetId
    readonly offerings: readonly {
      readonly configurationId: ModelServingConfigurationId
      readonly providerModelId: ProviderModelId
    }[]
  }[] },
  targetId: ModelOfferingTargetId,
  configurationId: ModelServingConfigurationId,
): Option.Option<ProviderModelId> => Option.fromNullable(
  models.models.find((model) => model.targetId === targetId)
    ?.offerings.find((offering) => offering.configurationId === configurationId)
    ?.providerModelId,
)

/**
 * Selects one exact assessed local configuration as durable slot intent.
 * Existing offerings are reused; otherwise ACN creates the offering before
 * the returned provider-model identity is assigned to the requested slot.
 */
export function useLocalConfigurationSelection() {
  const client = useAgentClient()
  const modelsAtom = useMirroredStateAtom(LocalModelsMirror)
  const createOfferingAtom = useMemo(
    () => client.mutation("CreateLocalModelOffering"),
    [client],
  )
  const assignAtom = useMemo(() => client.mutation("AssignSlot"), [client])
  const selectAtom = useMemo(
    () => Atom.fn<LocalConfigurationSelection>()((selection, get) => Effect.gen(function* () {
      const snapshot = Result.value(get(modelsAtom))
      const existing = Option.flatMap(snapshot, ({ state }) =>
        findLocalConfigurationOffering(
          state,
          selection.targetId,
          selection.configurationId,
        ))
      const providerModelId = Option.isSome(existing)
        ? existing.value
        : yield* get.setResult(createOfferingAtom, {
            payload: { configurationId: selection.configurationId },
            reactivityKeys: [LocalModelsMirror.id, ProviderModelCatalogMirror.id],
          })
      yield* get.setResult(assignAtom, {
        payload: {
          slotId: selection.slotId,
          selection: {
            providerId: ProviderIdSchema.make("local"),
            providerModelId,
            reasoningEffort: selection.reasoningEffort,
          },
        },
        reactivityKeys: [ModelSlotsMirror.id],
      })
      return providerModelId
    })),
    [assignAtom, createOfferingAtom, modelsAtom],
  )
  const result = useAtomValue(selectAtom)
  const select = useAtomSet(selectAtom)
  return { result, select }
}

export function useModelSlotActions() {
  const client = useAgentClient()
  const assignAtom = useMemo(() => client.mutation("AssignSlot"), [client])
  const clearAtom = useMemo(() => client.mutation("ClearSlot"), [client])
  const loadAtom = useMemo(() => client.mutation("LoadModel"), [client])
  const stopAtom = useMemo(() => client.mutation("StopModel"), [client])
  const favoriteAtom = useMemo(() => client.mutation("SetModelFavorite"), [client])
  const assignResult = useAtomValue(assignAtom)
  const clearResult = useAtomValue(clearAtom)
  const loadResult = useAtomValue(loadAtom)
  const stopResult = useAtomValue(stopAtom)
  const favoriteResult = useAtomValue(favoriteAtom)
  const assign = useAtomSet(assignAtom)
  const clear = useAtomSet(clearAtom)
  const load = useAtomSet(loadAtom)
  const stop = useAtomSet(stopAtom)
  const favorite = useAtomSet(favoriteAtom)

  return {
    assignResult,
    clearResult,
    loadResult,
    stopResult,
    favoriteResult,
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
