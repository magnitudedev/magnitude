import { Option } from "effect"
import type { ProviderModelId } from "@magnitudedev/ai/provider/model"
import {
  installedAcquisition,
  type CatalogIdentity,
  type CatalogModelReconciliationAdmission,
  type LocalModel,
  type LocalModelsState,
  type ModelCapabilities,
} from "./model-state"

/**
 * Pure projections over `LocalModelsState` shared by the contract's mutation
 * synchronization and by client presentation.
 */

export const localModelCapabilities = (
  model: LocalModel,
): Option.Option<ModelCapabilities> => model.servingState._tag === "Assessed"
  ? Option.some(model.servingState.capabilities)
  : Option.none()

export const localModelProviderModelId = (
  model: LocalModel,
): Option.Option<ProviderModelId> => {
  if (model.servingState._tag !== "Assessed") return Option.none()
  const availabilityState = model.servingState.availabilityState
  if (availabilityState._tag === "Installable") return Option.none()
  if (availabilityState._tag === "Unavailable") return availabilityState.providerModelId
  return Option.some(availabilityState.providerModelId)
}

export const localModelCatalogIdentity = (
  model: LocalModel,
): Option.Option<CatalogIdentity> => model.catalogMembershipState._tag === "InCatalog"
  ? Option.some({
    modelId: model.catalogMembershipState.catalogData.modelId,
    variantId: model.catalogMembershipState.catalogData.variantId,
  })
  : Option.none()

export const findLocalModelById = (
  models: readonly LocalModel[],
  modelId: ProviderModelId,
): Option.Option<LocalModel> => Option.fromNullable(models.find((model) => model.modelId === modelId))

export const findLocalModelByCatalogIdentity = (
  models: readonly LocalModel[],
  identity: CatalogIdentity,
): Option.Option<LocalModel> => Option.fromNullable(models.find((model) =>
  Option.exists(localModelCatalogIdentity(model), (candidate) =>
    candidate.modelId === identity.modelId && candidate.variantId === identity.variantId)))

export const installedLocalModels = (
  state: LocalModelsState,
): readonly LocalModel[] =>
  state.models.filter((model) => installedAcquisition(model.acquisitionState) !== undefined
    && model.servingState._tag === "Assessed"
    && model.servingState.assessment._tag === "Fits")

const admissionIsVisibleOn = (
  model: LocalModel,
  admission: CatalogModelReconciliationAdmission,
): boolean => {
  const acquisition = model.acquisitionState
  if (admission._tag === "Current") return acquisition._tag === "Installed"
  return acquisition._tag === "Installing"
    || acquisition._tag === "Updating"
    || acquisition._tag === "Installed"
}

/** The admitted reconciliation is visible on the canonical model. */
export const installationAdmissionIsVisible = (
  state: LocalModelsState,
  modelId: ProviderModelId,
  admission: CatalogModelReconciliationAdmission,
): boolean => Option.exists(
  findLocalModelById(state.models, modelId),
  (model) => admissionIsVisibleOn(model, admission),
)

/** The admitted reconciliation is visible on the model identified by catalog identity. */
export const catalogReconciliationIsVisible = (
  state: LocalModelsState,
  identity: CatalogIdentity,
  admission: CatalogModelReconciliationAdmission,
): boolean => Option.exists(
  findLocalModelByCatalogIdentity(state.models, identity),
  (model) => admissionIsVisibleOn(model, admission),
)
