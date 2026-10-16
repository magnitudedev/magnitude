import { Option } from "effect"
import type { ProviderModelId } from "@magnitudedev/ai/provider/model"
import {
  installedAcquisition,
  type CatalogSupport,
  type CustomModelAcquisitionState,
  type CustomModelOrigin,
  type DrafterCandidate,
  type LocalModel,
  type LocalModelAcquisitionState,
  type LocalModelsState,
  type ModelCapabilities,
  type ModelInstallation,
  type ModelResidency,
  type LocalModelServingState,
  type ServingProfile,
} from "./model-state"

/**
 * Pure projections over `LocalModelsState` shared by the contract's mutation
 * synchronization and by client presentation.
 */

export const localModelServingState = (
  model: LocalModel,
): Option.Option<LocalModelServingState> => Option.some(model.servingState)

export const localModelServingProfile = (
  model: LocalModel,
): Option.Option<ServingProfile> => {
  const serving = model.servingState
  switch (serving._tag) {
    case "Assessed": return Option.some(serving.assessment.profile)
    case "Assessing": return Option.some(serving.profile)
    case "Failed": return Option.isOption(serving.profile) ? serving.profile : Option.some(serving.profile)
  }
}

export const localModelCapabilities = (
  model: LocalModel,
): Option.Option<ModelCapabilities> => Option.flatMap(localModelServingState(model), (serving) =>
  serving._tag === "Assessed"
    ? Option.some(serving.capabilities)
    : Option.none())

export type CatalogDeprecation = Extract<CatalogSupport, { readonly _tag: "Deprecated" }>

export const localModelIsAvailable = (model: LocalModel): boolean => model._tag !== "Catalog"
  || model.catalogData.support._tag === "Supported"

/** A deprecated catalog model is never recommended, installed, or loaded. */
export const localModelDeprecation = (
  model: LocalModel,
): Option.Option<CatalogDeprecation> => model._tag === "Catalog"
  && model.catalogData.support._tag === "Deprecated"
  ? Option.some(model.catalogData.support)
  : Option.none()

/** The managed lifecycle of a catalog or added model; imported and found models have none. */
export const localModelManagedAcquisition = (
  model: LocalModel,
): Option.Option<LocalModelAcquisitionState> => {
  const state: CustomModelAcquisitionState = model.acquisitionState
  return state._tag === "External" || state._tag === "Unavailable" ? Option.none() : Option.some(state)
}

const localModelInstallation = (model: LocalModel): Option.Option<ModelInstallation> => {
  const state: CustomModelAcquisitionState = model.acquisitionState
  if (state._tag === "External" || state._tag === "Unavailable") return Option.some(state.installation)
  return Option.map(Option.fromNullable(installedAcquisition(state)), ({ installation }) => installation)
}

/** The runtime residency of a model whose files are present and readable. */
export const localModelResidency = (model: LocalModel): Option.Option<ModelResidency> => {
  const state: CustomModelAcquisitionState = model.acquisitionState
  if (state._tag === "External") return Option.some(state.residencyState)
  if (state._tag === "Unavailable") return Option.none()
  return Option.map(Option.fromNullable(installedAcquisition(state)), ({ residencyState }) => residencyState)
}

export const localModelProviderModelId = (
  model: LocalModel,
): Option.Option<ProviderModelId> => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  if (serving?._tag !== "Assessed" || serving.assessment._tag !== "Fits") return Option.none()
  if (!localModelIsAvailable(model)) return Option.none()
  if (model.acquisitionState._tag === "External") return Option.some(model.modelId)
  return Option.exists(localModelManagedAcquisition(model), (state) =>
    installedAcquisition(state) !== undefined && state._tag !== "Removing")
    ? Option.some(model.modelId)
    : Option.none()
}

export const findLocalModelById = (
  models: readonly LocalModel[],
  modelId: ProviderModelId,
): Option.Option<LocalModel> => Option.fromNullable(models.find((model) => model.modelId === modelId))

export const localModelInstalledBytes = (model: LocalModel): Option.Option<number> =>
  Option.map(localModelInstallation(model), ({ installedBytes }) => installedBytes)

export const localModelStorageBytes = (model: LocalModel): Option.Option<number> =>
  Option.orElse(localModelInstalledBytes(model), () => Option.some(model.storageBytes))

export const localModelIsInstalled = (model: LocalModel): boolean => Option.isSome(localModelInstallation(model))

export const installedLocalModels = (
  state: LocalModelsState,
): readonly LocalModel[] =>
  state.models.filter(localModelIsInstalled)

/** The tag naming where a custom model's files came from. */
export const customModelSourceLabel = (origin: CustomModelOrigin): string => {
  switch (origin._tag) {
    case "Added": return "Hugging Face"
    case "Imported": return "GGUF file"
    case "Found":
      switch (origin.source._tag) {
        case "HuggingFaceCache": return "Hugging Face cache"
        case "LmStudio": return "LM Studio"
        case "LlamaCpp": return "llama.cpp"
        case "Ollama": return "Ollama"
        case "Folder": return origin.source.path.split(/[\\/]/).filter((part) => part.length > 0).at(-1)
          ?? origin.source.path
      }
  }
}

/** The preselected drafter: the first candidate, in order, that the engine binds. */
export const preselectedDrafterCandidate = (
  candidates: readonly DrafterCandidate[],
): Option.Option<DrafterCandidate> => Option.fromNullable(candidates.find(({ verdict }) => verdict._tag === "Bound"))
