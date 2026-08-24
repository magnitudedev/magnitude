import { Option } from "effect"
import type {
  LocalModel,
  LocalModelRecommendation,
  LocalModelsState,
  ModelSlotsState,
} from "@magnitudedev/sdk"
import { servableModelBundlePackages } from "@magnitudedev/sdk"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import {
  installedLocalModels,
  localModelProviderModelId,
} from "./projection"

export interface LocalModelOption {
  readonly id: string
  readonly kind: "running" | "stored" | "recommendation"
  readonly model: LocalModel
  readonly recommendations: readonly LocalModelRecommendation[]
}

const kindOrder: Record<LocalModelOption["kind"], number> = {
  running: 0,
  stored: 1,
  recommendation: 2,
}

const intentOrder = {
  balanced: 0,
  smartest: 1,
  fastest: 2,
  lightweight: 3,
} as const

const orderedRecommendations = (
  recommendations: readonly LocalModelRecommendation[],
): readonly LocalModelRecommendation[] => [...recommendations]
  .filter((recommendation, index, all) => all.findIndex(({ intent }) =>
    intent === recommendation.intent) === index)
  .sort((left, right) => intentOrder[left.intent] - intentOrder[right.intent])

export const localModelBundleKey = (model: LocalModel): string =>
  model.bundle._tag === "Standalone"
    ? `package:${model.bundle.package.id}`
    : `speculative:${model.bundle.method._tag}:${model.bundle.draftSource._tag}:${servableModelBundlePackages(model.bundle)
      .map(({ id }) => id)
      .join(":")}`

export const localModelOptions = (
  models: LocalModelsState,
  slots: ModelSlotsState,
): readonly LocalModelOption[] => {
  const recommendationsByBundle = new Map<string, LocalModelRecommendation[]>()
  for (const model of models.models) {
    if (model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits") continue
    const key = localModelBundleKey(model)
    recommendationsByBundle.set(key, [
      ...(recommendationsByBundle.get(key) ?? []),
      ...model.servingState.recommendations,
    ])
  }
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
    recommendations: orderedRecommendations(
      recommendationsByBundle.get(localModelBundleKey(model)) ?? [],
    ),
  }))
  const representedBundles = new Set(installed.map(({ model }) => localModelBundleKey(model)))
  const recommendations = models.models.flatMap((model): readonly LocalModelOption[] => {
    if (representedBundles.has(localModelBundleKey(model))
      || model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits") return []
    const recommendations = orderedRecommendations(model.servingState.recommendations)
    const acquisitionActive = model.acquisitionState._tag === "Downloading"
      || model.acquisitionState._tag === "Failed"
    if (recommendations.length === 0 && !acquisitionActive) return []
    return [{
      id: recommendations.length === 0
        ? `acquisition:${model.modelId}`
        : `recommendation:${recommendations[0]!.id}`,
      kind: "recommendation",
      model,
      recommendations,
    }]
  })
  return [...installed, ...recommendations].sort((left, right) =>
    kindOrder[left.kind] - kindOrder[right.kind]
    || (left.kind === "recommendation" && right.kind === "recommendation"
      ? (left.recommendations[0] ? intentOrder[left.recommendations[0].intent] : 4)
        - (right.recommendations[0] ? intentOrder[right.recommendations[0].intent] : 4)
      : 0)
    || formatLocalModelDisplayName(left.model).localeCompare(formatLocalModelDisplayName(right.model))
    || left.id.localeCompare(right.id))
}
