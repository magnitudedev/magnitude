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
  ReasoningEfforts,
  ProviderModelAvailability,
  ProviderModelDisabledReason,
  ProviderId,
  ProviderModelId,
  ModelFamilyId,
} from "./model"
export type { ProviderModelCapabilities } from "../model/capabilities"
export type { ModelCatalog, ModelCatalogConfig } from "./catalog"
export type {
  Provider,
  ProviderModelBindOptions,
  RequestAttribution,
  WebSearchExtension,
  WebSearchResult,
  UsageExtension,
  UsageQuery,
  UsageResponse,
} from "./contract"
export type { BaseCallOptions, ToolChoice } from "./call-options"

// Errors
export { ModelCatalogError } from "./catalog"

// Functions
export { AVAILABLE_PROVIDER_MODEL, isProviderModelAvailable } from "./model"
export {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ModelFamilyIdSchema,
  ModelPricingInfoSchema,
  ProviderModelDisabledReasonSchema,
  ProviderModelAvailabilitySchema,
  ProviderModelSchema,
  ProviderModelFields,
  ReasoningEffortSchema,
  ReasoningEffortsSchema,
  VisionProperty,
  ReasoningProperty,
} from "./model"
export {
  defineModelDiscoverableProperty,
  ModelDiscoveryOperationIdSchema,
  ModelDiscoveryPhaseSchema,
  ModelPropertyDiscoveryErrorSchema,
  ModelPropertyDiscoveryRequestSchema,
  ModelPropertyNameSchema,
} from "./discoverable-property"
export type {
  ModelDiscoveryOperationId,
  ModelDiscoveryPhase,
  ModelPropertyDiscoveryError,
  ModelPropertyDiscoveryRequest,
  ModelPropertyName,
} from "./discoverable-property"
export { ProviderModelCapabilitiesSchema } from "../model/capabilities"
export { makeFileBackedModelCatalog } from "./file-catalog"
