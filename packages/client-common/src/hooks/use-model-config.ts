/**
 * Model config hook — shared between web, desktop, and CLI.
 *
 * Derives everything from the `GetCachedModelList` RPC (models, slot profiles,
 * and user config all in one response).
 */
import { useMemo } from "react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { SLOT_IDS, type SlotId } from "@magnitudedev/sdk"
import type { ModelSummary, ProviderInfo } from "@magnitudedev/sdk"

export interface ModelOption {
  providerId: string
  providerModelId: string
  modelFamilyId: string
  displayName: string
  slots?: readonly string[]
  contextWindow: number
  maxOutputTokens: number
  capabilities: { vision: boolean }
  reasoningEfforts: readonly string[]
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
  updateSlotModel: (slotId: SlotId, providerId: string | null, providerModelId: string | null) => void
  /** Update a slot's reasoning effort (null clears override) */
  updateSlotReasoning: (slotId: SlotId, effort: string | null) => void
  /** Reset all overrides to defaults */
  resetToDefaults: () => void
  /** Refresh the cached model list from the provider */
  refreshModels: () => void
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
    capabilities: { vision: model.capabilities.vision },
    reasoningEfforts: model.reasoningEfforts,
    pricing: model.pricing ?? null,
  }
}

export function useModelConfig(): UseModelConfigResult {
  const client = useAgentClient()
  const queryAtom = useMemo(
    () => client.query("GetCachedModelList", {}, { reactivityKeys: ["modelConfig"] }),
    [client],
  )
  const result = useAtomValue(queryAtom)

  const modelsLoading = Result.isInitial(result)
  const modelsError = Result.isFailure(result) ? "Failed to load available models" : null

  const data = Result.match(result, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (success) => success.value,
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

  const updateConfigAtom = useMemo(() => client.mutation("UpdateModelConfig"), [client])
  const refreshMutationAtom = useMemo(() => client.mutation("RefreshCachedModelList"), [client])
  const updateConfigResult = useAtomValue(updateConfigAtom)
  const refreshMutationResult = useAtomValue(refreshMutationAtom)
  const updateConfig = useAtomSet(updateConfigAtom)
  const refreshMutation = useAtomSet(refreshMutationAtom)

  const updateSlotModel = useMemo(
    () => (slotId: SlotId, providerId: string | null, providerModelId: string | null): void => {
      const current = configResponse?.slots[slotId] ?? {}
      updateConfig({
        payload: {
          slots: {
            [slotId]: {
              providerId: providerId ?? undefined,
              providerModelId: providerModelId ?? undefined,
              reasoningEffort: current.reasoningEffort,
            },
          },
        },
        reactivityKeys: ["modelConfig"],
      })
    },
    [updateConfig, configResponse],
  )

  const updateSlotReasoning = useMemo(
    () => (slotId: SlotId, effort: string | null): void => {
      const current = configResponse?.slots[slotId] ?? {}
      updateConfig({
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
    () => (): void => {
      updateConfig({
        payload: {
          slots: {},
        },
        reactivityKeys: ["modelConfig"],
      })
    },
    [updateConfig],
  )

  const refreshModels = useMemo(
    () => (): void => {
      refreshMutation({
        payload: {},
        reactivityKeys: ["modelConfig"],
      })
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
    updating: Result.isWaiting(updateConfigResult),
    updateError: Result.isFailure(updateConfigResult) ? "Failed to update model configuration" : null,
    refreshingModels: Result.isWaiting(refreshMutationResult),
    refreshModelsError: Result.isFailure(refreshMutationResult) ? "Failed to refresh models" : null,
  }
}
