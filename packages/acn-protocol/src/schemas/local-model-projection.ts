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
  type LocalModelAssessment,
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

/** The profile an assessment was made for; an unreadable model's has none. */
export const localModelAssessmentProfile = (
  assessment: LocalModelAssessment,
): Option.Option<ServingProfile> => assessment._tag === "Unsupported"
  ? assessment.profile
  : Option.some(assessment.profile)

/**
 * The profile a model's assessment was made for, the context clients show. Until the model is
 * assessed (still downloading, or assessing) its serving state names only the profile it will be
 * assessed for, a placeholder for a model whose files aren't read yet, so there is none to show.
 */
export const localModelServingProfile = (
  model: LocalModel,
): Option.Option<ServingProfile> => model.servingState._tag === "Assessed"
  ? localModelAssessmentProfile(model.servingState.assessment)
  : Option.none()

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

/** The model an ID names: its own ID, or a catalog model's alias (its files' ID as a found model). */
export const findLocalModelById = (
  models: readonly LocalModel[],
  modelId: ProviderModelId,
): Option.Option<LocalModel> => Option.fromNullable(models.find((model) => model.modelId === modelId)
  ?? models.find((model) => model._tag === "Catalog" && model.catalogData.aliases.some((alias) => alias === (modelId as string))))

/** The canonical ID for an ID that may be a catalog model's alias; any other ID is its own. */
export const canonicalLocalModelId = <Id extends ProviderModelId>(
  models: readonly LocalModel[],
  modelId: Id,
): Id | LocalModel["modelId"] => Option.match(
  Option.fromNullable(models.find((model) => model._tag === "Catalog"
    && model.catalogData.aliases.some((alias) => alias === (modelId as string)))),
  { onNone: () => modelId, onSome: (model) => model.modelId },
)

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

/**
 * The preselected drafter: the first candidate, in ICN's order, that automatic choice may take. It
 * binds, the user didn't enter it, and it never makes a model that fits without a drafter not fit.
 */
export const preselectedDrafterCandidate = (
  candidates: readonly DrafterCandidate[],
): Option.Option<DrafterCandidate> => Option.fromNullable(candidates.find(({ automatic }) => automatic))
