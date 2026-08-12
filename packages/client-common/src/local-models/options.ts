import { Option } from "effect"
import type {
  LocalModel,
  LocalModelRecommendation,
  LocalModelsState,
  ModelSlotsState,
} from "@magnitudedev/sdk"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import {
  installedLocalModels,
  localModelConfigurationId,
  localModelProviderModelId,
} from "./projection"

export interface LocalModelOption {
  readonly id: string
  readonly kind: "running" | "stored" | "recommendation"
  readonly model: LocalModel
  readonly recommendation: Option.Option<LocalModelRecommendation>
}

const kindOrder: Record<LocalModelOption["kind"], number> = {
  running: 0,
  stored: 1,
  recommendation: 2,
}

const intentOrder = {
  balanced: 0,
  best_quality: 1,
  fastest: 2,
  lightweight: 3,
} as const

export const localModelBundleKey = (model: LocalModel): string =>
  model.bundle._tag === "Standalone"
    ? `package:${model.bundle.package.id}`
    : `speculative-pair:${model.bundle.target.id}:${model.bundle.draft.id}`

export const localModelOptions = (
  models: LocalModelsState,
  slots: ModelSlotsState,
): readonly LocalModelOption[] => {
  const runningProviderModelIds = new Set([
    slots.slots.primary,
    slots.slots.secondary,
  ].flatMap((slot) => slot._tag === "ConfiguredLocal"
    && Option.exists(slot.instance, (instance) => instance.lifecycle._tag === "Ready")
    ? [slot.selection.providerModelId]
    : []))
  const installed = installedLocalModels(models).map((model): LocalModelOption => ({
    id: `installed:${Option.getOrElse(
      localModelConfigurationId(model),
      () => localModelBundleKey(model),
    )}`,
    kind: Option.exists(localModelProviderModelId(model), (providerModelId) =>
      runningProviderModelIds.has(providerModelId)) ? "running" : "stored",
    model,
    recommendation: Option.none(),
  }))
  const representedBundles = new Set(installed.map(({ model }) => localModelBundleKey(model)))
  const recommendations = models.models.flatMap((model): readonly LocalModelOption[] => {
    if (representedBundles.has(localModelBundleKey(model))
      || model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits") return []
    const recommendation = [...model.servingState.recommendations].sort((left, right) =>
      intentOrder[left.intent] - intentOrder[right.intent])[0]
    const acquisitionActive = model.acquisitionState._tag === "Downloading"
      || model.acquisitionState._tag === "Failed"
    if (recommendation === undefined && !acquisitionActive) return []
    return [{
      id: recommendation === undefined
        ? `acquisition:${model.servingState.configuration.id}`
        : `recommendation:${recommendation.id}`,
      kind: "recommendation",
      model,
      recommendation: Option.fromNullable(recommendation),
    }]
  })
  return [...installed, ...recommendations].sort((left, right) =>
    kindOrder[left.kind] - kindOrder[right.kind]
    || (left.kind === "recommendation" && right.kind === "recommendation"
      ? (Option.isSome(left.recommendation) ? intentOrder[left.recommendation.value.intent] : 4)
        - (Option.isSome(right.recommendation) ? intentOrder[right.recommendation.value.intent] : 4)
      : 0)
    || formatLocalModelDisplayName(left.model).localeCompare(formatLocalModelDisplayName(right.model))
    || left.id.localeCompare(right.id))
}
