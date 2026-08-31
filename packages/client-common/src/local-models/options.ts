import { Option } from "effect"
import type {
  LocalInferenceHardware,
  LocalModel,
  LocalModelRankingScores,
  LocalModelsState,
  ModelSlotsState,
} from "@magnitudedev/sdk"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import { installedLocalModels, localModelProviderModelId } from "./projection"

export interface LocalModelOption {
  readonly id: string
  readonly kind: "running" | "stored" | "downloadable"
  readonly model: LocalModel
}

export interface LocalModelRankingPreference {
  readonly fastToSmart: number
  readonly memoryBudgetBytes: number
}

export const LOCAL_MODEL_RANKING_SCALE_LABELS = [
  "Fastest",
  "Faster",
  "Balanced",
  "Smarter",
  "Smartest",
] as const

export const LOCAL_MODEL_RANKING_SCALE_VALUES = [0.05, 0.25, 0.5, 0.75, 0.95] as const

export const LOCAL_MODEL_RANKING_SCALE_INTERVALS = LOCAL_MODEL_RANKING_SCALE_VALUES.length - 1

export const localModelRankingScaleIndex = (value: number): number =>
  LOCAL_MODEL_RANKING_SCALE_VALUES.reduce((closestIndex, candidate, index) =>
    Math.abs(candidate - value) < Math.abs(LOCAL_MODEL_RANKING_SCALE_VALUES[closestIndex]! - value)
      ? index
      : closestIndex, 0)

const clamp01 = (value: number): number => Number.isFinite(value)
  ? Math.min(1, Math.max(0, value))
  : 0

const kindOrder: Record<LocalModelOption["kind"], number> = {
  running: 0,
  stored: 1,
  downloadable: 2,
}

export const localModelRankingUtility = (
  scores: LocalModelRankingScores,
  fastToSmart: number,
): number => {
  const preference = clamp01(fastToSmart)
  return scores.intelligence ** (0.9 * preference)
    * scores.speed ** (0.9 * (1 - preference))
    * scores.fidelity ** 0.1
}

export const targetPhysicalMemoryBytes = (hardware: LocalInferenceHardware): number =>
  hardware.memoryDomains.reduce((total, domain) => total + domain.totalBytes, 0)

export const rankedLocalModelOptions = (
  options: readonly LocalModelOption[],
  preference: LocalModelRankingPreference,
  limit = 10,
): readonly LocalModelOption[] => options
  .flatMap((option): readonly { readonly option: LocalModelOption; readonly utility: number }[] => {
    if (option.model._tag !== "Catalog") return []
    const serving = option.model.servingState
    if (serving._tag !== "Assessed"
      || serving.assessment._tag !== "Fits"
      || !("rankingScores" in serving)
      || Option.isNone(serving.rankingScores)
      || !Number.isFinite(preference.memoryBudgetBytes)
      || serving.assessment.memory.totalRequiredBytes > Math.max(0, preference.memoryBudgetBytes)) return []
    return [{
      option,
      utility: localModelRankingUtility(serving.rankingScores.value, preference.fastToSmart),
    }]
  })
  .sort((left, right) => right.utility - left.utility
    || left.option.model.modelId.localeCompare(right.option.model.modelId))
  .slice(0, Math.max(0, Math.floor(limit)))
  .map(({ option }) => option)

export const localModelOptions = (
  models: LocalModelsState,
  slots: ModelSlotsState,
): readonly LocalModelOption[] => {
  const runningProviderModelIds = new Set([
    slots.slots.primary,
    slots.slots.secondary,
  ].flatMap((slot) => slot._tag === "ConfiguredLocal"
    && slot.residency._tag === "Ready"
    ? [slot.selection.providerModelId]
    : []))
  const installed = installedLocalModels(models).map((model): LocalModelOption => ({
    id: `installed:${model.modelId}`,
    kind: Option.exists(localModelProviderModelId(model), (providerModelId) =>
      runningProviderModelIds.has(providerModelId)) ? "running" : "stored",
    model,
  }))
  const representedModelIds = new Set(installed.map(({ model }) => model.modelId))
  const downloadable = models.models.flatMap((model): readonly LocalModelOption[] => {
    if (model._tag !== "Catalog"
      || representedModelIds.has(model.modelId)
      || model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits"
      || !("rankingScores" in model.servingState)
      || Option.isNone(model.servingState.rankingScores)) return []
    return [{
      id: `downloadable:${model.modelId}`,
      kind: "downloadable",
      model,
    }]
  })
  return [...installed, ...downloadable].sort((left, right) =>
    kindOrder[left.kind] - kindOrder[right.kind]
    || formatLocalModelDisplayName(left.model).localeCompare(formatLocalModelDisplayName(right.model))
    || left.id.localeCompare(right.id))
}
