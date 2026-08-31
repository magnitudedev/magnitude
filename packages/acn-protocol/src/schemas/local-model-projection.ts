import { Option } from "effect"
import type { ProviderModelId } from "@magnitudedev/ai/provider/model"
import {
  installedAcquisition,
  type LocalModel,
  type LocalModelsState,
  type ModelCapabilities,
  type LocalModelServingState,
  type ServingProfile,
} from "./model-state"

/**
 * Pure projections over `LocalModelsState` shared by the contract's mutation
 * synchronization and by client presentation.
 */

export const localModelServingState = (
  model: LocalModel,
): Option.Option<LocalModelServingState> => model._tag === "Catalog"
  ? Option.some(model.servingState)
  : model.state._tag === "Ready"
    ? Option.some(model.state.servingState)
    : Option.none()

export const localModelServingProfile = (
  model: LocalModel,
): Option.Option<ServingProfile> => {
  if (model._tag === "Discovered") {
    if (model.state._tag !== "Ready") return Option.none()
    const serving = model.state.servingState
    return serving._tag === "Assessed"
      ? Option.some(serving.assessment.profile)
      : Option.some(serving.profile)
  }
  const serving = model.servingState
  return serving._tag === "Assessed"
    ? Option.some(serving.assessment.profile)
    : serving._tag === "Assessing"
      ? Option.some(serving.profile)
      : serving.profile
}

export const localModelCapabilities = (
  model: LocalModel,
): Option.Option<ModelCapabilities> => Option.flatMap(localModelServingState(model), (serving) =>
  serving._tag === "Assessed"
    ? Option.some(serving.capabilities)
    : Option.none())

export const localModelProviderModelId = (
  model: LocalModel,
): Option.Option<ProviderModelId> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed" || serving.assessment._tag !== "Fits") return Option.none()
  if (model._tag === "Discovered") return Option.some(model.modelId)
  return installedAcquisition(model.acquisitionState) !== undefined
      && model.acquisitionState._tag !== "Removing"
    ? Option.some(model.modelId)
    : Option.none()
}

export const findLocalModelById = (
  models: readonly LocalModel[],
  modelId: ProviderModelId,
): Option.Option<LocalModel> => Option.fromNullable(models.find((model) => model.modelId === modelId))

export const localModelInstalledBytes = (model: LocalModel): Option.Option<number> => {
  if (model._tag === "Discovered") {
    return Option.some(model.state.installation.installedBytes)
  }
  return Option.fromNullable(installedAcquisition(model.acquisitionState)?.installation.installedBytes)
}

export const localModelStorageBytes = (model: LocalModel): Option.Option<number> =>
  Option.orElse(localModelInstalledBytes(model), () => model._tag === "Catalog"
    ? Option.some(model.storageBytes)
    : Option.none())

export const localModelIsInstalled = (model: LocalModel): boolean =>
  model._tag === "Discovered"
    ? true
    : installedAcquisition(model.acquisitionState) !== undefined

export const installedLocalModels = (
  state: LocalModelsState,
): readonly LocalModel[] =>
  state.models.filter(localModelIsInstalled)
