// Classifier
export {
  type Atom,
  type AtomType,
  atomizeModelId,
  isAllDigits,
} from "./classifier/atomizer"
export {
  type PatternSymbol,
  lit,
  sep,
  dot,
  num,
  ver,
  opt,
} from "./classifier/symbols"
export {
  type ClassifyResult,
  type Family,
  type PatternEntry,
  classify,
} from "./classifier/classify"
export {
  MODEL_FAMILIES,
  getModelFamily,
  classifyModelFamily,
  classifyModelFamilyFromEvidence,
  FAMILY_DEFINITIONS,
} from "./family-registry"

// Registry & aggregation
export {
  ProviderRegistry,
  type ProviderRegistryService,
  type DiscoverableProviderInstance,
  type ProviderInfo,
  type AuthStatus,
  makeProviderRegistry,
  ProviderRegistryLive,
} from "./registry"
export {
  makeAggregatedCatalog,
  inspectProviderCatalogs,
  type ProviderCatalogOutcome,
  buildFamilies,
} from "./catalog-aggregator"

// Magnitude provider
export {
  createMagnitudeProvider,
  fetchUsage,
  PROVIDER_ID as MAGNITUDE_PROVIDER_ID,
  type MagnitudeProviderInstance,
  type MagnitudeClientConfig,
  type FetchUsageOptions,
  WebSearchError,
  MagnitudeClientError,
} from "./magnitude/provider"
export type { WebSearchResult, UsageQuery } from "@magnitudedev/ai"
export { createMagnitudeCatalog, toMagnitudeModelInfo, type MagnitudeAuthentication } from "./magnitude/catalog"
export { MagnitudeModelListResponseSchema, MagnitudeRawModelSchema } from "./magnitude/contract"
export {
  createMagnitudeCompatibleSpec,
  type MagnitudeCallOptions,
  type MagnitudeModelSpec,
  type MagnitudeCompatibleSpecConfig,
} from "./magnitude/models"
export {
  classifyMagnitudeRejectedResponse,
  tryParseErrorBody,
  type ParsedMagnitudeApiError,
} from "./magnitude/errors"
export type {
  MagnitudeModelInfo,
  MagnitudeRawModel,
  ModelListResponse,
  MagnitudeAdditionalOptions,
  MagnitudeApiError,
  MagnitudeErrorType,
  MagnitudeErrorCode,
  MagnitudeErrorDetails,
  UsageLimitDetails,
  SubscriptionRequiredDetails,
  BillingWindowBudget,
  BillingWindowName,
  ProSubscriptionStatus,
  ReasoningEffort,
  ModelPricingInfo,
} from "./magnitude/contract"
export type { ToolChoice } from "@magnitudedev/ai"
export type { CloudUsageResponse, UsagePeriod } from "./magnitude/usage"
