import { useCallback, useMemo } from "react"
import { Option, type Equivalence } from "effect"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  type CatalogFormModelId,
  type LocalModelsState,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import {
  LocalModels,
  useLocalModelMutations,
} from "../local-models/service"
import { ModelSlots, useModelSlotMutations } from "../model-slots/service"
import { providerCatalogFromCatalog } from "../model-catalog/projection"

export const useLocalInferenceHardware = () => {
  const client = useAgentClient()
  const hardware = useMemo(() => Atom.make((get) =>
    get(client.Models.GetLocalEnvironment({})).result), [client])
  return useAtomValue(hardware)
}
export type LocalInferenceHardwareResult = ReturnType<typeof useLocalInferenceHardware>

export const useLocalModels = () => {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(LocalModels), [client])
  const state = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (models) => get(models.state))), [service])
  return useAtomValue(state)
}

export const useLocalModelsSelector = <Selection,>(
  selector: (state: LocalModelsState) => Selection,
  equivalent: Equivalence.Equivalence<Selection>,
) => {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(LocalModels), [client])
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
  const service = useMemo(() => client.runtime.atom(LocalModels), [client])
  const catalog = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (models) => get(models.catalog))), [service])
  return useAtomValue(catalog)
}

export const useModelSlots = () => {
  const client = useAgentClient()
  const service = useMemo(() => client.runtime.atom(ModelSlots), [client])
  const state = useMemo(() => Atom.make((get) =>
    Result.flatMap(get(service), (slots) => get(slots.state))), [service])
  return useAtomValue(state)
}

export const useProviderModelCatalog = () => {
  const client = useAgentClient()
  const catalog = useMemo(() => Atom.make((get) =>
    Result.map(get(client.Models.GetCatalog({})).result, providerCatalogFromCatalog)), [client])
  return useAtomValue(catalog)
}

/**
 * The advisory load plan for a slot. It is not poked by name: it rereads
 * whenever the hardware or slot snapshot it depends on changes.
 */
export function usePreviewModelLoad(slotId: SlotId) {
  const client = useAgentClient()
  const modelSlots = useMemo(() => client.runtime.atom(ModelSlots), [client])
  const preview = useMemo(() => {
    return Atom.make((get) => {
      get(client.Models.GetLocalEnvironment({}))
      return Result.flatMap(get(modelSlots), (service) =>
        Result.flatMap(get(service.state), (state) => {
          const slot = slotId === "primary" ? state.slots.primary : state.slots.secondary
          return slot._tag === "ConfiguredLocal"
            ? get(client.Models.PreviewSlotLoad({ slotId })).result
            : Result.initial()
        }))
    })
  }, [client, modelSlots, slotId])
  return useAtomValue(preview)
}

export function useLocalModelActions() {
  const client = useAgentClient()
  const mutations = useLocalModelMutations()
  const selectLocalModel = useAtomSet(client.Models.SelectLocalModel)

  return {
    ...mutations,
    installAndAssign: useCallback((
      modelId: CatalogFormModelId,
      slotId: SlotId,
      reasoningEffort: SlotSelection["reasoningEffort"],
    ) => {
      selectLocalModel({ modelId, slotId, reasoningEffort })
    }, [selectLocalModel]),
  }
}

export function useModelSlotActions() {
  return useModelSlotMutations()
}
