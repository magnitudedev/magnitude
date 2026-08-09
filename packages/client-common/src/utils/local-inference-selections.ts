import { Option } from "effect"
import {
  ProviderModelCatalogLifecycle,
  ProviderIdSchema,
  type LocalModel,
  type LocalModelCatalogCandidate,
  type LocalModelRecommendation,
  type LocalModelsState,
  type ModelServingConfigurationId,
  type ModelSlotsState,
  type ProviderModelCatalogState,
  type ProviderModelId,
  type ReasoningEffort,
} from "@magnitudedev/sdk"

type LocalInferenceSelectionBase = {
  readonly id: string
  readonly model: LocalModel
  readonly contextLength: number
  readonly providerModelId: Option.Option<ProviderModelId>
  readonly reasoningEffort: Option.Option<ReasoningEffort>
}

export type LocalInferenceSelection = LocalInferenceSelectionBase & (
  | {
      readonly kind: "running" | "stored"
      readonly configurationId: ModelServingConfigurationId
      readonly recommendation: { readonly _tag: "None" }
    }
  | {
      readonly kind: "recommendation"
      readonly recommendation:
        | { readonly _tag: "None" }
        | { readonly _tag: "Recommended"; readonly value: LocalModelRecommendation }
    }
)

const selectionKindOrder: Record<LocalInferenceSelection["kind"], number> = {
  running: 0,
  stored: 1,
  recommendation: 2,
}

const recommendationIntentOrder = {
  balanced: 0,
  best_quality: 1,
  fastest: 2,
  lightweight: 3,
} as const

const compareSelections = (
  left: LocalInferenceSelection,
  right: LocalInferenceSelection,
): number => selectionKindOrder[left.kind] - selectionKindOrder[right.kind]
  || (left.kind === "recommendation" && right.kind === "recommendation"
    ? (left.recommendation._tag === "Recommended"
      ? recommendationIntentOrder[left.recommendation.value.intent]
      : 4) - (right.recommendation._tag === "Recommended"
      ? recommendationIntentOrder[right.recommendation.value.intent]
      : 4)
    : 0)
  || left.model.displayName.localeCompare(right.model.displayName)

export const buildLocalInferenceSelections = (
  models: LocalModelsState,
  catalog: ProviderModelCatalogState,
  slots: ModelSlotsState,
): readonly LocalInferenceSelection[] => {
  const localProviderId = ProviderIdSchema.make("local")
  const running = new Set([slots.slots.primary, slots.slots.secondary].flatMap((slot) =>
    slot._tag === "ConfiguredLocal"
      && Option.exists(slot.instance, (instance) => instance.lifecycle._tag === "Ready")
      ? [slot.selection.providerModelId]
      : []))
  const catalogModels = ProviderModelCatalogLifecycle.match(catalog, {
    Loading: () => [],
    Ready: ({ models }) => models,
    Refreshing: ({ models }) => models,
    Degraded: ({ models }) => models,
    Unavailable: () => [],
  })
  const localProviderIds = new Set(catalogModels
    .filter(({ providerId, availability }) =>
      providerId === localProviderId && availability._tag === "Available")
    .map(({ providerModelId }) => providerModelId))
  const installedCandidates = models.recommendations._tag === "Ready"
    ? models.recommendations.catalog.reduce((selected, candidate) => {
        if (candidate.download._tag !== "Downloaded"
          || candidate.availability._tag !== "Available") return selected
        if (!selected.has(candidate.targetId)) selected.set(candidate.targetId, candidate)
        return selected
      }, new Map<string, LocalModelCatalogCandidate>())
    : new Map<string, LocalModelCatalogCandidate>()
  const installed = models.models.flatMap((model): readonly LocalInferenceSelection[] => {
    if (model.download._tag !== "Downloaded") return []
    const candidate = installedCandidates.get(model.targetId)
    const availableOfferings = model.offerings.filter(({ providerModelId }) =>
      localProviderIds.has(providerModelId))
    const offering = availableOfferings.find(({ providerModelId }) => running.has(providerModelId))
      ?? availableOfferings.find(({ configurationId }) =>
        configurationId === candidate?.configurationId)
      ?? (candidate === undefined ? availableOfferings[0] : undefined)
    if (offering === undefined && candidate === undefined) return []
    const providerModelId = Option.fromNullable(offering?.providerModelId)
    const configurationId = offering?.configurationId ?? candidate!.configurationId
    const providerModel = offering === undefined
      ? undefined
      : catalogModels.find(({ providerModelId }) => providerModelId === offering.providerModelId)
    return [{
      id: `installed:${model.targetId}`,
      kind: Option.exists(providerModelId, (id) => running.has(id)) ? "running" : "stored",
      model,
      configurationId,
      recommendation: { _tag: "None" },
      providerModelId,
      contextLength: providerModel?.contextWindow
        ?? candidate?.profile.contextLength
        ?? model.maximumContextLength,
      reasoningEffort: providerModel?.capabilities.reasoning.defaultEffort
        ?? candidate?.capabilities.reasoning.defaultEffort
        ?? Option.none(),
    }]
  })
  const recommendations = models.recommendations._tag === "Ready"
    ? models.recommendations.entries.flatMap((recommendation): readonly LocalInferenceSelection[] => {
        const model = models.models.find(({ targetId }) =>
          targetId === recommendation.candidate.targetId)
        if (!model || model.download._tag === "Downloaded") return []
        return [{
          id: `recommendation:${recommendation.id}`,
          kind: "recommendation",
          model,
          recommendation: { _tag: "Recommended", value: recommendation },
          contextLength: recommendation.candidate.profile.contextLength,
          providerModelId: Option.none(),
          reasoningEffort: recommendation.candidate.capabilities.reasoning.defaultEffort,
        }]
      })
    : []
  const representedModelIds = new Set(recommendations.map(({ model }) => model.targetId))
  const transientDownloads = models.models
    .filter((model) =>
      (model.download._tag === "Downloading" || model.download._tag === "Failed")
      && !representedModelIds.has(model.targetId))
    .map((model): LocalInferenceSelection => ({
      id: `download:${model.targetId}`,
      kind: "recommendation",
      model,
      recommendation: { _tag: "None" },
      contextLength: model.maximumContextLength,
      providerModelId: Option.fromNullable(model.offerings[0]?.providerModelId),
      reasoningEffort: Option.none(),
    }))
  return [...installed, ...recommendations, ...transientDownloads].sort(compareSelections)
}
