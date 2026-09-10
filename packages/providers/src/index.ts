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
  MagnitudeClientError,
} from "./magnitude/provider"
export {
  createExaWebSearch,
  type ExaWebSearchConfig,
  type ExaWebSearchInstance,
} from "./exa/web-search"
export {
  WebSearchProviderSchema,
  WebSearchNotConfigured,
  WebSearchRequestEncodingFailed,
  WebSearchRequestFailed,
  WebSearchTimedOut,
  WebSearchRejected,
  WebSearchResponseReadFailed,
  WebSearchInvalidResponse,
  formatWebSearchError,
  type WebSearchProvider,
  type WebSearchError,
} from "./web-search-error"
export type { WebSearchResult, UsageQuery } from "@magnitudedev/ai"
export { createMagnitudeCatalog, toMagnitudeModelInfo, type MagnitudeAuthentication } from "./magnitude/catalog"
export {
  MagnitudeAdditionalOptionsSchema,
  MagnitudeModelListResponseSchema,
  MagnitudeRawModelSchema,
  TurnConstraintsSchema,
} from "./magnitude/contract"
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

// MiniMax provider
export {
  createMiniMaxProvider,
  MINIMAX_ENDPOINTS,
  MINIMAX_MODELS,
  MINIMAX_PROVIDER_ID,
  createMiniMaxCatalog,
  createMiniMaxCompatibleSpec,
  classifyMiniMaxRejectedResponse,
  type MiniMaxAuthentication,
  type MiniMaxCallOptions,
  type MiniMaxClientConfig,
  type MiniMaxCompatibleSpecConfig,
  type MiniMaxEndpointConfig,
  type MiniMaxModelInfo,
  type MiniMaxProviderInstance,
  type MiniMaxRegion,
} from "./minimax/index"

// Custom OpenAI-compatible endpoints
export {
  createCustomEndpointProvider,
  customEndpointProviderId,
} from "./custom-endpoint/provider"
