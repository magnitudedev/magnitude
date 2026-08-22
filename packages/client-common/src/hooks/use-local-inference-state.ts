import { useCallback, useMemo } from "react"
import { Effect, Option, type Equivalence } from "effect"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  GetModelSlots,
  LocalInferenceHardwareMirror,
  ProviderIdSchema,
  ProviderModelCatalogMirror,
  type LocalModelsState,
  type ModelServingConfigurationId,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import {
  LocalModels,
  useLocalModelMutations,
} from "../local-models/service"
import { ModelSlots, useModelSlotMutations } from "../model-slots/service"
import { useMirroredState } from "./use-mirrored-state"

export const useLocalInferenceHardware = () =>
  Result.map(useMirroredState(LocalInferenceHardwareMirror), ({ state }) => state)
export type LocalInferenceHardwareResult = ReturnType<typeof useLocalInferenceHardware>

export const useLocalModels = () => {
  const client = useAgentClient()
  const service = useMemo(() => client.effectQuery.runtime.atom(LocalModels), [client])
  const state = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (models) => get(models.state))), [service])
  return useAtomValue(state)
}

export const useLocalModelsSelector = <Selection,>(
  selector: (state: LocalModelsState) => Selection,
  equivalent: Equivalence.Equivalence<Selection>,
) => {
  const client = useAgentClient()
  const service = useMemo(() => client.effectQuery.runtime.atom(LocalModels), [client])
  const selection = useMemo(() => {
    let previous = Option.none<Selection>()
    return Atom.make((get) => Option.map(
      Result.value(Result.flatMap(get(service), (models) => get(models.state))),
      (models) => {
        const next = selector(models)
        if (Option.isSome(previous) && equivalent(previous.value, next)) return previous.value
        previous = Option.some(next)
        return next
      },
    ))
  }, [equivalent, selector, service])
  return useAtomValue(selection)
}

export const useCatalogModels = () => {
  const client = useAgentClient()
  const service = useMemo(() => client.effectQuery.runtime.atom(LocalModels), [client])
  const catalog = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (models) => get(models.catalog))), [service])
  return useAtomValue(catalog)
}

export const useModelSlots = () => {
  const client = useAgentClient()
  const service = useMemo(() => client.effectQuery.runtime.atom(ModelSlots), [client])
  const state = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (slots) => get(slots.state))), [service])
  return Result.map(useAtomValue(state), ({ state: slots }) => slots)
}

export const useProviderModelCatalog = () =>
  Result.map(useMirroredState(ProviderModelCatalogMirror), ({ state }) => state)

export function usePreviewModelLoad(slotId: SlotId) {
  const client = useAgentClient()
  const preview = useMemo(
    () => client.rpc.query(
      "PreviewModelLoad",
      { slotId },
      { reactivityKeys: [LocalInferenceHardwareMirror.id, GetModelSlots.name] },
    ),
    [client, slotId],
  )
  return useAtomValue(preview)
}

export function useLocalModelActions() {
  const client = useAgentClient()
  const mutations = useLocalModelMutations()
  const installAndAssignAction = useMemo(() => Atom.keepAlive(client.effectQuery.runtime.fn<{
    readonly configurationId: ModelServingConfigurationId
    readonly slotId: SlotId
    readonly reasoningEffort: SlotSelection["reasoningEffort"]
  }>()(({ configurationId, slotId, reasoningEffort }) => Effect.flatMap(
    LocalModels,
    (models) => models.install(configurationId),
  ).pipe(
    Effect.flatMap(({ providerModelId }) => Effect.flatMap(
      ModelSlots,
      (slots) => slots.assign(slotId, {
        providerId: ProviderIdSchema.make("local"),
        providerModelId,
        reasoningEffort,
      }),
    )),
  ))), [client])
  const installAndAssign = useAtomSet(installAndAssignAction)

  return {
    ...mutations,
    installAndAssign: useCallback((
      configurationId: ModelServingConfigurationId,
      slotId: SlotId,
      reasoningEffort: SlotSelection["reasoningEffort"],
    ) => {
      installAndAssign({ configurationId, slotId, reasoningEffort })
    }, [installAndAssign]),
  }
}

export function useModelSlotActions() {
  return useModelSlotMutations()
}
