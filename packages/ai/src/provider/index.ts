// Provider-agnostic contract — defines what a provider is.
// No specific provider IDs, no specific model families, no specific model IDs.
// Concrete family lists and classifiers live in packages/providers.

// Types
export type {
  ProviderModel,
  ModelFamily,
  ModelFamilyCapabilities,
  ModelPricingInfo,
  ReasoningEffort,
  ProviderModelAvailability,
  ProviderModelDisabledReason,
  ProviderId,
  ProviderModelId,
  ModelFamilyId,
} from "./model"
export type { ProviderModelCapabilities } from "../model/capabilities"
export type { ModelProfile } from "./profile"
export type { ModelCatalog, ModelCatalogConfig } from "./catalog"
export type {
  Provider,
  ProviderModelBindOptions,
  WebSearchExtension,
  WebSearchResult,
  BalanceExtension,
  BalanceQuery,
  BalanceResponse,
} from "./contract"
export type { BaseCallOptions, ToolChoice } from "./call-options"

// Errors
export { ModelCatalogError } from "./catalog"

// Functions
export { toModelProfile } from "./profile"
export { AVAILABLE_PROVIDER_MODEL, isProviderModelAvailable } from "./model"
export {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ModelFamilyIdSchema,
  ModelPricingInfoSchema,
  ProviderModelDisabledReasonSchema,
  ProviderModelAvailabilitySchema,
  ProviderModelSchema,
} from "./model"
export { ProviderModelCapabilitiesSchema } from "../model/capabilities"
export { makeFileBackedModelCatalog } from "./file-catalog"
