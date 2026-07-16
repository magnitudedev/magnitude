/**
 * Model config hook — shared between web, desktop, and CLI.
 *
 * Composes the independent reactive model-catalog and model-slot resources.
 */
import { useMemo } from "react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { useAgentClient } from "../state/agent-client-context"
import { ModelSlotsLifecycle, type SlotId } from "@magnitudedev/sdk"
import { useModelCatalog, useModelSlots } from "./use-reactive-rpc"

export function useModelConfig() {
  const client = useAgentClient()
  const catalogResult = useModelCatalog()
  const slotsResult = useModelSlots()

  const slotConfiguration = Option.flatMap(Result.value(slotsResult), ({ state }) => ModelSlotsLifecycle.match(state, {
    loading: () => Option.none(),
    ready: ({ config }) => Option.some(config),
    refreshing: ({ config }) => Option.some(config),
    degraded: ({ config }) => Option.some(config),
    unavailable: ({ config }) => Option.some(config),
  }))

  // ── Mutations ──

  const updateConfigAtom = useMemo(() => client.mutation("UpdateModelSlots"), [client])
  const refreshMutationAtom = useMemo(() => client.mutation("RefreshModelCatalog"), [client])
  const updateConfigResult = useAtomValue(updateConfigAtom)
  const refreshMutationResult = useAtomValue(refreshMutationAtom)
  const updateConfig = useAtomSet(updateConfigAtom)
  const refreshMutation = useAtomSet(refreshMutationAtom)

  const updateSlotModel = useMemo(
    () => (slotId: SlotId, providerId: string | null, providerModelId: string | null): void => {
      const current = Option.flatMap(slotConfiguration, (config) => Option.fromNullable(config.slots[slotId]))
      updateConfig({
        payload: {
          slots: {
            [slotId]: {
              providerId: providerId ?? undefined,
              providerModelId: providerModelId ?? undefined,
              reasoningEffort: Option.getOrUndefined(Option.map(current, (entry) => entry.reasoningEffort)),
            },
          },
        },
        reactivityKeys: ["modelSlots"],
      })
    },
    [updateConfig, slotConfiguration],
  )

  const updateSlotReasoning = useMemo(
    () => (slotId: SlotId, effort: string | null): void => {
      const current = Option.flatMap(slotConfiguration, (config) => Option.fromNullable(config.slots[slotId]))
      updateConfig({
        payload: {
          slots: {
            [slotId]: {
              providerId: Option.getOrUndefined(Option.map(current, (entry) => entry.providerId)),
              providerModelId: Option.getOrUndefined(Option.map(current, (entry) => entry.providerModelId)),
              reasoningEffort: effort ?? undefined,
            },
          },
        },
        reactivityKeys: ["modelSlots"],
      })
    },
    [updateConfig, slotConfiguration],
  )

  const resetToDefaults = useMemo(
    () => (): void => {
      updateConfig({
        payload: {
          slots: {},
        },
        reactivityKeys: ["modelSlots"],
      })
    },
    [updateConfig],
  )

  const refreshModels = useMemo(
    () => (): void => {
      refreshMutation({
        payload: { providerId: Option.none() },
        reactivityKeys: ["modelCatalog", "modelSlots"],
      })
    },
    [refreshMutation],
  )

  return {
    catalog: catalogResult,
    slots: slotsResult,
    slotUpdate: updateConfigResult,
    catalogRefresh: refreshMutationResult,
    updateSlotModel,
    updateSlotReasoning,
    resetToDefaults,
    refreshModels,
  }
}

export type UseModelConfigResult = ReturnType<typeof useModelConfig>
