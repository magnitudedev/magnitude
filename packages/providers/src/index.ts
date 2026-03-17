/**
 * @magnitudedev/providers — Multi-provider LLM support with model slot abstraction.
 */

// Types
export type {
  BamlProviderType,
  AuthFlowType,
  AuthMethodDef,
  ModelDefinition,
  ProviderDefinition,
  AuthInfo,
  ApiKeyAuth,
  OAuthAuth,
  AwsAuth,
  GcpAuth,
  ModelSelection,
  MagnitudeConfig,
  ProviderOptions,
} from './types'

// Model types
export type { ModelDriverId, ModelDriver } from './model/model-driver'
export { DRIVERS } from './model/model-driver'
export { Model, type ModelCosts } from './model/model'
export { ModelConnection, type ModelConnection as ModelConnectionType } from './model/model-connection'
export type { InferenceConfig } from './model/inference-config'
export type { BoundModel, ChatStream, ModelFunctionDef, StreamingFn, CompleteFn } from './model/bound-model'

// Model functions
export {
  CodingAgentChat,
  SimpleChat,
  CodingAgentCompact,
  GenerateChatTitle,
  ExtractMemoryDiff,
  GatherSplit,
  PatchFile,
  CreateFile,
  AutopilotContinuation,
} from './model/model-function'

// Errors
export {
  NotConfigured,
  ProviderDisconnected,
  AuthFailed,
  ContextLimitExceeded,
  RateLimited,
  TransportError,
  ParseError,
} from './errors/model-error'
export type { ModelError } from './errors/model-error'
export { classifyHttpError, classifyUnknownError } from './errors/classify-error'

// Legacy singleton state (compatibility — prefer runtime DI surface for new code)
export type { ModelSlot, CallUsage, SlotUsage } from './state/provider-state'
export { peekSlot, getModelContextWindow } from './state/provider-state'

// Resolver (Effect services)
export { ModelResolver } from './resolver/model-resolver'
export { makeModelResolver } from './resolver/model-resolver-live'
export { makeTestResolver } from './resolver/model-runtime-test'
export type { TestModelConfig } from './resolver/model-runtime-test'
export { TraceEmitter, TracePersister, makeTracePersister, makeNoopTracer, makeTestTracer } from './resolver/tracing'
export type { TraceData } from './resolver/tracing'

// Runtime DI surface
export { ProviderCatalog, ProviderState, ProviderAuth } from './runtime/contracts'
export { makeProviderRuntimeLive } from './runtime/live'
export { bootstrapProviderRuntime } from './runtime/bootstrap'
export { createProviderClient } from './runtime/client'
export type { ProviderClient } from './runtime/client'

// Registry
export { PROVIDERS, getProvider, getProviderIds, getStaticProviderModels, setProviderModels, getModelCost } from './registry'

// Catalog
export * from './catalog'

// Detection
export { detectProviders, detectDefaultProvider, detectProviderAuthMethods } from './detect'
export type { DetectedProvider, DetectedAuthMethod, ProviderAuthMethodStatus } from './detect'

// ClientRegistry builder
export { buildClientRegistry } from './client-registry-builder'

// Recommendations
export {
  getModelRecommendation,
  normalizeModelId,
  resolveRecommendedModel,
  MODEL_RECOMMENDATION_RULES,
} from './model-recommendations'
export type { ModelRecommendationRule, RecommendationMatch } from './model-recommendations'

// Reasoning effort
export { getLowestEffortOptions } from './reasoning-effort'

// Usage calculation
export { buildUsage, calculateCosts } from './usage'

// Output normalization
export { normalizeModelOutput, normalizeQuotesInString } from './util/output-normalization'

// Drivers
export { BamlDriver } from './drivers/baml-driver'
export { ResponsesDriver } from './drivers/responses-driver'

// Browser-compatible models
export {
  BROWSER_COMPATIBLE_MODELS,
  isBrowserCompatible,
  getBrowserCompatibleModels,
  detectBrowserModel,
} from './browser-models'

// OAuth flows
export {
  startAnthropicOAuth,
  exchangeAnthropicCode,
  refreshAnthropicToken,
  ANTHROPIC_OAUTH_BETA_HEADERS,
} from './auth/anthropic-oauth'
export type { AnthropicOAuthStart } from './auth/anthropic-oauth'

export {
  startOpenAIBrowserOAuth,
  startOpenAIDeviceOAuth,
  refreshOpenAIToken,
} from './auth/openai-oauth'
export type { OpenAIBrowserOAuthStart, OpenAIDeviceOAuthStart } from './auth/openai-oauth'

export {
  startCopilotAuth,
  exchangeCopilotToken,
  COPILOT_HEADERS,
} from './auth/copilot-oauth'
export type { CopilotOAuthStart } from './auth/copilot-oauth'