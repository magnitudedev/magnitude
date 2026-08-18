import { Option } from "effect"
import type {
  LocalModel,
  LocalModelsState,
  ModelCapabilities,
  ModelServingConfigurationId,
  ProviderModelId,
} from "@magnitudedev/sdk"

export const localModelCapabilities = (
  model: LocalModel,
): Option.Option<ModelCapabilities> => model.servingState._tag === "Assessed"
  ? Option.some(model.servingState.capabilities)
  : Option.none()

export const localModelConfigurationId = (
  model: LocalModel,
): Option.Option<ModelServingConfigurationId> => {
  const servingState = model.servingState
  if (servingState._tag === "Resolving") return Option.none()
  if (servingState._tag === "Failed") {
    return Option.map(servingState.configuration, ({ id }) => id)
  }
  return Option.some(servingState.configuration.id)
}

export const localModelProviderModelId = (
  model: LocalModel,
): Option.Option<ProviderModelId> => {
  if (model.servingState._tag !== "Assessed") return Option.none()
  const availabilityState = model.servingState.availabilityState
  if (availabilityState._tag === "Installable") return Option.none()
  if (availabilityState._tag === "Unavailable") return availabilityState.providerModelId
  return Option.some(availabilityState.providerModelId)
}

export const findLocalModelByConfigurationId = (
  models: readonly LocalModel[],
  configurationId: ModelServingConfigurationId,
): Option.Option<LocalModel> => Option.fromNullable(models.find((model) =>
  Option.contains(localModelConfigurationId(model), configurationId)))

export const installedLocalModels = (
  state: LocalModelsState,
): readonly LocalModel[] =>
  state.models.filter((model) => model.acquisitionState._tag === "Installed"
    && model.servingState._tag === "Assessed"
    && model.servingState.assessment._tag === "Fits")
