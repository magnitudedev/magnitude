/**
 * Model config hook — shared between web, desktop, and CLI.
 *
 * Composes the independent reactive model-catalog and model-slot resources.
 */
import { useMemo } from "react"
import { useAtomValue, useAtomSet, Result } from "@effect-atom/atom-react"
import { Option } from "effect"
import { useAgentClient } from "../state/agent-client-context"
import { SLOT_IDS, type SlotId } from "@magnitudedev/sdk"
import type { ModelSummary, ProviderInfo, ProviderModelAvailability } from "@magnitudedev/sdk"
import { useModelCatalog, useModelSlots } from "./use-reactive-rpc"

export interface ModelOption {
  providerId: string
  providerModelId: string
  modelFamilyId?: string
  displayName: string
  slots?: readonly string[]
  contextWindow: number
  maxOutputTokens: number
  capabilities: { vision?: boolean }
  availability: ProviderModelAvailability
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
    availability: model.availability,
    reasoningEfforts: model.reasoningEfforts,
    pricing: model.pricing ?? null,
  }
}

export function useModelConfig(): UseModelConfigResult {
  const client = useAgentClient()
  const catalogResult = useModelCatalog()
  const slotsResult = useModelSlots()

  const modelsLoading = Result.isInitial(catalogResult)
  const modelsError = Result.isFailure(catalogResult) ? "Failed to load available models" : null

  const catalog = Result.match(catalogResult, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (success) => success.value,
  })
  const slots = Result.match(slotsResult, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (success) => success.value,
  })
  const catalogLoading = modelsLoading || (catalog?.refreshing === true && catalog.models.length === 0)

  const models = useMemo(() => {
    if (!catalog) return null
    return catalog.models.map(toModelOption)
  }, [catalog])

  const configResponse = slots?.config ?? null
  const providers = catalog?.providers ?? null

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

  const updateConfigAtom = useMemo(() => client.mutation("UpdateModelSlots"), [client])
  const refreshMutationAtom = useMemo(() => client.mutation("RefreshModelCatalog"), [client])
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
        reactivityKeys: ["modelSlots"],
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
        reactivityKeys: ["modelSlots"],
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
        reactivityKeys: ["modelSlots"],
      })
    },
    [updateConfig],
  )

  const refreshModels = useMemo(
    () => (): void => {
      refreshMutation({
        payload: { providerId: Option.none() },
        reactivityKeys: ["modelCatalog"],
      })
    },
    [refreshMutation],
  )

  return {
    models,
    providers,
    modelsLoading: catalogLoading,
    modelsError,
    slotConfig,
    updateSlotModel,
    updateSlotReasoning,
    resetToDefaults,
    refreshModels,
    updating: Result.isWaiting(updateConfigResult),
    updateError: Result.isFailure(updateConfigResult) ? "Failed to update model configuration" : null,
    refreshingModels: Result.isWaiting(refreshMutationResult) || catalog?.refreshing === true,
    refreshModelsError: Result.isFailure(refreshMutationResult) ? "Failed to refresh models" : null,
  }
}
