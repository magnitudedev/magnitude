import { Option } from "effect"
import type {
  LocalModel,
  LocalModelCatalogCandidate,
  LocalModelsState,
  ModelServingConfigurationId,
  ModelSlotsState,
  ProviderModelCatalogEntry,
  ReasoningEffort,
  ModelCapabilities,
} from "@magnitudedev/sdk"

export const localModelCapabilities = (
  model: LocalModel,
): Option.Option<ModelCapabilities> => model.readiness._tag === "Assessed"
  ? Option.some(model.readiness.capabilities)
  : Option.none()

export interface InstalledLocalModelChoice {
  readonly id: string
  readonly model: LocalModel
  readonly configurationId: ModelServingConfigurationId
  readonly candidate: Option.Option<LocalModelCatalogCandidate>
  readonly providerModel: Option.Option<ProviderModelCatalogEntry>
  readonly running: boolean
  readonly contextLength: number
  readonly reasoningEffort: Option.Option<ReasoningEffort>
}

export const findLocalOffering = (
  models: readonly LocalModel[],
  configurationId: ModelServingConfigurationId,
): Option.Option<ProviderModelCatalogEntry> => Option.firstSomeOf(models.map((model) =>
  model.readiness._tag === "Assessed" && model.readiness.configuration.id === configurationId
    ? model.readiness.offering
    : Option.none()))

export const buildInstalledLocalModelChoices = (
  models: LocalModelsState,
  slots?: ModelSlotsState,
): readonly InstalledLocalModelChoice[] => {
  const running = new Set(slots === undefined ? [] : [
    slots.slots.primary,
    slots.slots.secondary,
  ].flatMap((slot) => slot._tag === "ConfiguredLocal"
    && Option.exists(slot.instance, (instance) => instance.lifecycle._tag === "Ready")
    ? [slot.selection.providerModelId]
    : []))
  const installedCandidates = new Map(
    models.recommendations._tag === "Ready"
      ? models.recommendations.catalog
        .filter(({ download }) => download._tag === "Downloaded")
        .map((candidate) => [candidate.configurationId, candidate] as const)
      : [],
  )

  return models.models.flatMap((model): readonly InstalledLocalModelChoice[] => {
    if (model.readiness._tag !== "Assessed" || model.readiness.assessment._tag !== "Fits") return []
    const { configuration, offering, capabilities } = model.readiness
    const providerModel = Option.getOrUndefined(offering)
    const candidate = installedCandidates.get(configuration.id)
    return [{
      id: `installed:${configuration.id}`,
      model,
      configurationId: configuration.id,
      candidate: Option.fromNullable(candidate),
      providerModel: Option.fromNullable(providerModel),
      running: providerModel !== undefined && running.has(providerModel.providerModelId),
      contextLength: providerModel?.contextWindow
        ?? candidate?.profile.contextLength
        ?? configuration.profile.contextLength,
      reasoningEffort: providerModel?.capabilities.reasoning.defaultEffort
        ?? candidate?.capabilities.reasoning.defaultEffort
        ?? capabilities.reasoning.defaultEffort,
    }]
  })
}
