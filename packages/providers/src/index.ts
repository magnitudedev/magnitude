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
  type LlamaCppProviderInstance,
  type LlamaCppModelInfo,
  type LlamaCppCallOptions,
  type LlamaCppToolChoice,
  type LlamaCppProviderSource,
  type LlamaCppInferenceLease,
  createLlamaCppProvider,
  createLlamaCppCompatibleSpec,
  classifyLlamaCppRejectedResponse,
  LlamaCppAcquisitionError,
  PROVIDER_ID as LLAMACPP_PROVIDER_ID,
} from "./llamacpp"
export {
  makeAggregatedCatalog,
  buildFamilies,
} from "./catalog-aggregator"

// Magnitude provider
export {
  createMagnitudeProvider,
  fetchBalance,
  PROVIDER_ID as MAGNITUDE_PROVIDER_ID,
  type MagnitudeProviderInstance,
  type MagnitudeClientConfig,
  type FetchBalanceOptions,
  WebSearchError,
  MagnitudeClientError,
} from "./magnitude/provider"
export type { WebSearchResult, BalanceQuery } from "@magnitudedev/ai"
export { createMagnitudeCatalog, toMagnitudeModelInfo } from "./magnitude/catalog"
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
  InsufficientCreditsDetails,
  ReasoningEffort,
  ModelPricingInfo,
} from "./magnitude/contract"
export type { ToolChoice } from "@magnitudedev/ai"
export type { BalanceResponse, UsagePeriod } from "./magnitude/usage"
