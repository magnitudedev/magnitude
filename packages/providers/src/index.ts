// Classifier
export {
  type Atom,
  type AtomType,
  atomizeModelId,
  isAllDigits,
} from "./classifier/atomizer"
export {
  SUPPORTED_PROVIDER_DEFINITIONS,
  getSupportedProviderDefinition,
  type SupportedProviderDefinition,
  type ProviderAuthKind,
} from "./definitions"
export {
  createModelsDevClient,
  type ModelsDevClient,
  type ModelsDevClientConfig,
  type ModelsDevModel,
  type ModelsDevOverride,
  type ModelsDevProvider,
  type ModelsDevSnapshot,
} from "./catalog/models-dev"
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
  type ConfiguredProviderInstance,
  type ProviderInfo,
  type AuthStatus,
  type AuthSource,
  makeProviderRegistry,
  ProviderRegistryLive,
} from "./registry"
export {
  type LlamaCppProviderInstance,
  type LlamaCppDiscoveryResult,
  type LlamaCppModelInfo,
  type LlamaCppCallOptions,
  type LlamaCppToolChoice,
  type LlamaCppRawModel,
  type LlamaCppModelMeta,
  type LlamaCppClientConfig,
  type ServerProps as LlamaCppServerProps,
  type ServerStatus as LlamaCppServerStatus,
  createLlamaCppProvider,
  createLlamaCppCatalog,
  createLlamaCppCompatibleSpec,
  classifyLlamaCppRejectedResponse,
  DEFAULT_LLAMACPP_ENDPOINT,
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
export { createMagnitudeCatalog } from "./magnitude/catalog"
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

export {
  createDeepSeekProvider,
  createDeepSeekCatalog,
  createDeepSeekCompatibleSpec,
  classifyDeepSeekRejectedResponse,
  DEFAULT_DEEPSEEK_ENDPOINT,
  type DeepSeekCallOptions,
  type DeepSeekClientConfig,
  type DeepSeekModelInfo,
  type DeepSeekProvider,
  type DeepSeekProviderInstance,
} from "./deepseek"
export {
  createKimiApiProvider,
  createKimiApiCatalog,
  createKimiApiCompatibleSpec,
  classifyKimiApiRejectedResponse,
  DEFAULT_KIMI_API_ENDPOINT,
  type KimiApiCallOptions,
  type KimiApiClientConfig,
  type KimiApiModelInfo,
  type KimiApiProvider,
  type KimiApiProviderInstance,
} from "./kimi-api"
export {
  createKimiForCodingProvider,
  createKimiForCodingCatalog,
  createKimiForCodingCompatibleSpec,
  classifyKimiForCodingRejectedResponse,
  DEFAULT_KIMI_FOR_CODING_ENDPOINT,
  KIMI_FOR_CODING_MODEL_ID,
  type KimiForCodingCallOptions,
  type KimiForCodingClientConfig,
  type KimiForCodingModelInfo,
  type KimiForCodingProvider,
  type KimiForCodingProviderInstance,
} from "./kimi-for-coding"
export {
  createZaiProvider,
  createZaiCatalog,
  createZaiCompatibleSpec,
  classifyZaiRejectedResponse,
  DEFAULT_ZAI_ENDPOINT,
  type ZaiCallOptions,
  type ZaiClientConfig,
  type ZaiModelInfo,
  type ZaiProvider,
  type ZaiProviderInstance,
} from "./zai"
export {
  createZaiCodingPlanProvider,
  createZaiCodingPlanCatalog,
  createZaiCodingPlanCompatibleSpec,
  classifyZaiCodingPlanRejectedResponse,
  DEFAULT_ZAI_CODING_PLAN_ENDPOINT,
  type ZaiCodingPlanCallOptions,
  type ZaiCodingPlanClientConfig,
  type ZaiCodingPlanModelInfo,
  type ZaiCodingPlanProvider,
  type ZaiCodingPlanProviderInstance,
} from "./zai-coding-plan"
export {
  createOpenRouterProvider,
  createOpenRouterCatalog,
  createOpenRouterCompatibleSpec,
  classifyOpenRouterRejectedResponse,
  DEFAULT_OPENROUTER_ENDPOINT,
  type OpenRouterCallOptions,
  type OpenRouterClientConfig,
  type OpenRouterModelInfo,
  type OpenRouterProvider,
  type OpenRouterProviderInstance,
} from "./openrouter"
export {
  createVercelProvider,
  createVercelCatalog,
  createVercelCompatibleSpec,
  classifyVercelRejectedResponse,
  DEFAULT_VERCEL_ENDPOINT,
  type VercelCallOptions,
  type VercelClientConfig,
  type VercelModelInfo,
  type VercelProvider,
  type VercelProviderInstance,
} from "./vercel"
