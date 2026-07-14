/**
 * Model config hook — shared between web, desktop, and CLI.
 *
 * Derives everything from the `GetCachedModelList` RPC (models, slot profiles,
 * and user config all in one response).
 */
import { useMemo, useState } from "react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { SLOT_IDS, type SlotId } from "@magnitudedev/sdk"
import type { ModelSummary, ModelConfigResponse, ModelList, ProviderInfo } from "@magnitudedev/sdk"

export interface ModelOption {
  providerId: string
  providerModelId: string
  modelFamilyId: string
  displayName: string
  slots?: readonly string[]
  contextWindow: number
  maxOutputTokens: number
  capabilities: ModelSummary["capabilities"]
  reasoningEfforts: readonly string[]
  openWeightStatus?: ModelSummary["openWeightStatus"]
  metadataSource?: ModelSummary["metadataSource"]
  description?: string
  upstreamFamily?: string
  modalities?: ModelSummary["modalities"]
  pricing: { input: number; output: number; cachedInput?: number } | null
}

export interface UseModelConfigResult {
  /** All available models from provider (same list for every slot — slot is just a default recommendation) */
  models: readonly ModelOption[] | null
  /** Provider connection and discovery status */
  providers: readonly ProviderInfo[] | null
  /** Whether models are loading */
  modelsLoading: boolean
  /** Models error message */
  modelsError: string | null
  /** Current user config per slot */
  slotConfig: Record<SlotId, { providerId?: string; providerModelId?: string; reasoningEffort?: string }> | null
  /** Update a slot's model (null clears override) */
  updateSlotModel: (slotId: SlotId, providerId: string | null, providerModelId: string | null) => Promise<void>
  /** Update a slot's reasoning effort (null clears override) */
  updateSlotReasoning: (slotId: SlotId, effort: string | null) => Promise<void>
  /** Reset all overrides to defaults */
  resetToDefaults: () => Promise<void>
  /** Refresh the cached model list from the provider */
  refreshModels: (providerId?: string) => Promise<void>
  /** Whether a mutation is in progress */
  updating: boolean
  /** Mutation error message */
  updateError: string | null
  /** Whether the refresh is in progress */
  refreshingModels: boolean
  /** Refresh error message */
  refreshModelsError: string | null
}

function toModelOption(model: ModelSummary): ModelOption {
  return {
    providerId: model.providerId,
    providerModelId: model.providerModelId,
    modelFamilyId: model.modelFamilyId,
    displayName: model.displayName,
    ...(model.slots ? { slots: model.slots } : {}),
    contextWindow: model.contextWindow,
    maxOutputTokens: model.maxOutputTokens,
    capabilities: model.capabilities,
    reasoningEfforts: model.reasoningEfforts,
    ...(model.openWeightStatus ? { openWeightStatus: model.openWeightStatus } : {}),
    ...(model.metadataSource ? { metadataSource: model.metadataSource } : {}),
    ...(model.description ? { description: model.description } : {}),
    ...(model.upstreamFamily ? { upstreamFamily: model.upstreamFamily } : {}),
    ...(model.modalities ? { modalities: model.modalities } : {}),
    pricing: model.pricing ?? null,
  }
}

export function useModelConfig(): UseModelConfigResult {
  const client = useAgentClient()
  const [refreshingModels, setRefreshingModels] = useState(false)
  const [refreshModelsError, setRefreshModelsError] = useState<string | null>(null)

  const result = useAtomValue(
    client.query("GetCachedModelList", {}, { reactivityKeys: ["modelConfig"] }),
  )

  const modelsLoading = Result.isInitial(result)
  const modelsError = Result.isFailure(result) ? "Failed to load available models" : null

  const data = Result.match(result, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (success) => success.value as ModelList,
  })

  const models = useMemo(() => {
    if (!data) return null
    return data.models.map(toModelOption)
  }, [data])

  const configResponse = data?.modelConfig ?? null
  const providers = data?.providers ?? null

  const slotConfig = useMemo(() => {
    if (!configResponse) return null
    const result: Record<SlotId, { providerId?: string; providerModelId?: string; reasoningEffort?: string }> = {
      primary: {},
      secondary: {},
    }
    for (const slotId of SLOT_IDS) {
      const entry = configResponse.slots[slotId]
      if (entry) {
        result[slotId] = {
          providerId: entry.providerId,
          providerModelId: entry.providerModelId,
          reasoningEffort: entry.reasoningEffort,
        }
      }
    }
    return result
  }, [configResponse])

  // ── Mutations ──

  const updateConfig = useAtomSet(
    client.mutation("UpdateModelConfig"),
    { mode: "promise" },
  )

  const refreshMutation = useAtomSet(
    client.mutation("RefreshCachedModelList"),
    { mode: "promise" },
  )

  const updateSlotModel = useMemo(
    () => async (slotId: SlotId, providerId: string | null, providerModelId: string | null): Promise<void> => {
      const current = configResponse?.slots[slotId] ?? {}
      const selectedModel = data?.models.find((model) =>
        model.providerId === providerId && model.providerModelId === providerModelId
      )
      const reasoningEffort = current.reasoningEffort
        && selectedModel?.reasoningEfforts.includes(current.reasoningEffort)
        ? current.reasoningEffort
        : undefined
      await updateConfig({
        payload: {
          slots: {
            [slotId]: {
              providerId: providerId ?? undefined,
              providerModelId: providerModelId ?? undefined,
              reasoningEffort,
            },
          },
        },
        reactivityKeys: ["modelConfig"],
      })
    },
    [updateConfig, configResponse, data],
  )

  const updateSlotReasoning = useMemo(
    () => async (slotId: SlotId, effort: string | null): Promise<void> => {
      const current = configResponse?.slots[slotId] ?? {}
      await updateConfig({
        payload: {
          slots: {
            [slotId]: {
              providerId: current.providerId,
              providerModelId: current.providerModelId,
              reasoningEffort: effort ?? undefined,
            },
          },
        },
        reactivityKeys: ["modelConfig"],
      })
    },
    [updateConfig, configResponse],
  )

  const resetToDefaults = useMemo(
    () => async (): Promise<void> => {
      await updateConfig({
        payload: {
          slots: {},
        },
        reactivityKeys: ["modelConfig"],
      })
    },
    [updateConfig],
  )

  const refreshModels = useMemo(
    () => async (providerId?: string): Promise<void> => {
      setRefreshingModels(true)
      setRefreshModelsError(null)
      try {
        await refreshMutation({
          payload: providerId ? { providerId } : {},
          reactivityKeys: ["modelConfig", "providerAuth"],
        })
      } catch (err) {
        setRefreshModelsError(err instanceof Error ? err.message : "Failed to refresh models")
        throw err
      } finally {
        setRefreshingModels(false)
      }
    },
    [refreshMutation],
  )

  return {
    models,
    providers,
    modelsLoading,
    modelsError,
    slotConfig,
    updateSlotModel,
    updateSlotReasoning,
    resetToDefaults,
    refreshModels,
    updating: false,
    updateError: null,
    refreshingModels,
    refreshModelsError,
  }
}
