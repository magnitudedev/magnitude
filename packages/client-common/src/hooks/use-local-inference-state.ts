import { useCallback, useMemo } from "react"
import { Effect, Option, type Equivalence } from "effect"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { QueryClient } from "@magnitudedev/effect-query"
import {
  Configuration,
  LocalInference,
  ProviderIdSchema,
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

export const useLocalInferenceHardware = () => {
  const client = useAgentClient()
  const hardware = useMemo(() => Atom.make((get) =>
    Result.map(get(client.query(LocalInference.GetLocalInferenceHardware, {})).result, ({ state }) => state)), [client])
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
  return Result.map(useAtomValue(state), ({ state: slots }) => slots)
}

export const useProviderModelCatalog = () => {
  const client = useAgentClient()
  const catalog = useMemo(() => Atom.make((get) =>
    Result.map(get(client.query(Configuration.GetProviderModelCatalog, {})).result, ({ state }) => state)), [client])
  return useAtomValue(catalog)
}

/**
 * The advisory load plan for a slot. It is not poked by name: it rereads
 * whenever the hardware or slot snapshot it depends on changes.
 */
export function usePreviewModelLoad(slotId: SlotId) {
  const client = useAgentClient()
  const refresh = useMemo(() => client.runtime.fn<void>()(
    () => QueryClient.invalidate(LocalInference.PreviewModelLoad.match({ slotId })),
  ), [client, slotId])
  const runRefresh = useAtomSet(refresh)
  const preview = useMemo(() => {
    let dependencies: readonly [number, number] | null = null
    return Atom.make((get) => {
      const hardware = get(client.query(LocalInference.GetLocalInferenceHardware, {}))
      const slots = get(client.query(Configuration.GetModelSlots, {}))
      const next: readonly [number, number] = [
        Option.getOrElse(hardware.dataUpdatedAt, () => 0),
        Option.getOrElse(slots.dataUpdatedAt, () => 0),
      ]
      if (dependencies !== null && (dependencies[0] !== next[0] || dependencies[1] !== next[1])) {
        queueMicrotask(() => runRefresh())
      }
      dependencies = next
      return get(client.query(LocalInference.PreviewModelLoad, { slotId })).result
    })
  }, [client, slotId, runRefresh])
  return useAtomValue(preview)
}

export function useLocalModelActions() {
  const client = useAgentClient()
  const mutations = useLocalModelMutations()
  const installAndAssignAction = useMemo(() => Atom.keepAlive(client.runtime.fn<{
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
