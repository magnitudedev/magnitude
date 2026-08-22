import { Option } from "effect"
import type { ProviderModelId } from "@magnitudedev/ai/provider/model"
import type {
  CatalogIdentity,
  CatalogModelReconciliationAdmission,
  LocalModel,
  LocalModelsState,
  ModelCapabilities,
  ModelServingConfigurationId,
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

export const localModelCatalogIdentity = (
  model: LocalModel,
): Option.Option<CatalogIdentity> => model.catalogMembershipState._tag === "InCatalog"
  ? Option.some({
    modelId: model.catalogMembershipState.catalogData.modelId,
    variantId: model.catalogMembershipState.catalogData.variantId,
  })
  : Option.none()

export const findLocalModelByConfigurationId = (
  models: readonly LocalModel[],
  configurationId: ModelServingConfigurationId,
): Option.Option<LocalModel> => Option.fromNullable(models.find((model) =>
  Option.contains(localModelConfigurationId(model), configurationId)))

export const findLocalModelByCatalogIdentity = (
  models: readonly LocalModel[],
  identity: CatalogIdentity,
): Option.Option<LocalModel> => Option.fromNullable(models.find((model) =>
  Option.exists(localModelCatalogIdentity(model), (candidate) =>
    candidate.modelId === identity.modelId && candidate.variantId === identity.variantId)))

export const installedLocalModels = (
  state: LocalModelsState,
): readonly LocalModel[] =>
  state.models.filter((model) => model.acquisitionState._tag === "Installed"
    && model.servingState._tag === "Assessed"
    && model.servingState.assessment._tag === "Fits")

const admissionIsVisibleOn = (
  model: LocalModel,
  admission: CatalogModelReconciliationAdmission,
): boolean => {
  const acquisition = model.acquisitionState
  const isCurrent = acquisition._tag === "Installed" && model.upgradeState._tag === "Current"
  if (admission._tag === "Current" || isCurrent) return isCurrent
  if (acquisition._tag !== "NotInstalled"
    && acquisition._tag !== "Installed"
    && acquisition.downloadId === admission.downloadId) return true
  return model.upgradeState._tag === "Upgrading"
    && model.upgradeState.downloadId === admission.downloadId
}

/** The admitted reconciliation is visible on the model identified by configuration. */
export const installationAdmissionIsVisible = (
  state: LocalModelsState,
  configurationId: ModelServingConfigurationId,
  admission: CatalogModelReconciliationAdmission,
): boolean => Option.exists(
  findLocalModelByConfigurationId(state.models, configurationId),
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
