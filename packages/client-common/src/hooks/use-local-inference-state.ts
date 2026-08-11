import { useCallback, useMemo } from "react"
import { Effect, Option, type Equivalence } from "effect"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Mutation, Query } from "@magnitudedev/effect-query"
import {
  LocalInferenceHardwareMirror,
  ModelSlotsMirror,
  ProviderModelCatalogMirror,
  ProviderIdSchema,
  type ModelInstanceId,
  type DownloadAttemptId,
  type LocalModelsState,
  type ModelServingConfigurationId,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { localModelAtoms } from "../local-models/atoms"
import { useMirroredState } from "./use-mirrored-state"

export const useLocalInferenceHardware = () =>
  Result.map(useMirroredState(LocalInferenceHardwareMirror), ({ state }) => state)
export type LocalInferenceHardwareResult = ReturnType<typeof useLocalInferenceHardware>

export const useLocalModelsResultAtom = () => {
  const client = useAgentClient()
  const atoms = useMemo(() => localModelAtoms(client), [client])
  useAtomMount(atoms.mirrorInvalidationWatchAtom)
  useAtomMount(atoms.invalidationBridgeAtom)
  return atoms.localModelsResultAtom
}

export const useLocalModels = () => useAtomValue(useLocalModelsResultAtom())

export const useLocalModelsSelector = <Selection,>(
  selector: (state: LocalModelsState) => Selection,
  equivalent: Equivalence.Equivalence<Selection>,
) => {
  const client = useAgentClient()
  const atoms = useMemo(() => localModelAtoms(client), [client])
  const selection = useMemo(() => Query.select(
    atoms.localModelsQuery(undefined),
    selector,
    equivalent,
  ), [atoms, equivalent, selector])
  useAtomMount(atoms.mirrorInvalidationWatchAtom)
  useAtomMount(atoms.invalidationBridgeAtom)
  return Result.value(useAtomValue(selection).result)
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
  const atoms = useMemo(() => localModelAtoms(client), [client])
  const installationMutationStates = useAtomValue(atoms.installationMutationStatesAtom)
  const install = useAtomSet(atoms.installMutation)
  const cancel = useAtomSet(atoms.cancelDownloadMutation)
  const dismiss = useAtomSet(atoms.dismissDownloadFailureMutation)
  const deleteModel = useAtomSet(atoms.deleteLocalModelMutation)
  const assignMutation = useMemo(() => client.mutation("AssignSlot"), [client])
  const installAndAssignAtom = useMemo(() => Atom.fn<{
    readonly configurationId: ModelServingConfigurationId
    readonly slotId: SlotId
    readonly reasoningEffort: SlotSelection["reasoningEffort"]
  }>()(({ configurationId, slotId, reasoningEffort }, get) => Effect.gen(function* () {
    const { providerModelId } = yield* Mutation.execute(atoms.installMutation, { configurationId })
    yield* get.setResult(assignMutation, {
      payload: {
        slotId,
        selection: {
          providerId: ProviderIdSchema.make("local"),
          providerModelId,
          reasoningEffort,
        },
      },
      reactivityKeys: [ModelSlotsMirror.id],
    })
  })), [assignMutation, atoms])
  const installAndAssign = useAtomSet(installAndAssignAtom)
  useAtomMount(atoms.mirrorInvalidationWatchAtom)
  useAtomMount(atoms.invalidationBridgeAtom)

  return {
    installationMutationStates,
    install: useCallback((configurationId: ModelServingConfigurationId) => {
      install({ configurationId })
    }, [install]),
    installAndAssign: useCallback((
      configurationId: ModelServingConfigurationId,
      slotId: SlotId,
      reasoningEffort: SlotSelection["reasoningEffort"],
    ) => {
      installAndAssign({ configurationId, slotId, reasoningEffort })
    }, [installAndAssign]),
    cancel: useCallback((attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]]) => {
      cancel({ attemptIds })
    }, [cancel]),
    dismissFailure: useCallback((attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]]) => {
      dismiss({ attemptIds })
    }, [dismiss]),
    delete: useCallback((configurationId: ModelServingConfigurationId) => {
      deleteModel({ configurationId })
    }, [deleteModel]),
  }
}

export function useModelSlotActions() {
  const client = useAgentClient()
  const assignAtom = useMemo(() => client.mutation("AssignSlot"), [client])
  const assignResult = useAtomValue(assignAtom)
  const assign = useAtomSet(assignAtom)
  const clear = useAtomSet(client.mutation("ClearSlot"))
  const load = useAtomSet(client.mutation("LoadModel"))
  const stop = useAtomSet(client.mutation("StopModel"))
  const favorite = useAtomSet(client.mutation("SetModelFavorite"))

  return {
    assignResult,
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
