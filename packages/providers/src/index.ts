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
export type { ModelSlot } from './state/provider-state'
export type { ModelDriverId, ModelDriver } from './model/model-driver'
export { DRIVERS } from './model/model-driver'
export { Model, type ModelCosts } from './model/model'
export { ModelConnection, type ModelConnection as ModelConnectionType } from './model/model-connection'
export type { InferenceConfig } from './model/inference-config'
export type { BoundModel, ChatStream, ModelFunctionDef, StreamingFn, CompleteFn } from './model/bound-model'

// Model functions
export {
  CodingAgentChat,
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

// State management
export type { CallUsage, SlotUsage } from './state/provider-state'
export {
  peekSlot,
  setModel,
  clearModel,
  initializeProviderState,
  getModelContextWindow,
  validateModelSwitch,
  getSlotUsage,
  resetSlotUsage,
  accumulateUsage,
} from './state/provider-state'


// Resolver (Effect services)
export { ModelResolver } from './resolver/model-resolver'
export type { ContextLimits } from './resolver/model-resolver'
export { makeModelResolver } from './resolver/model-resolver-live'
export { makeTestResolver } from './resolver/model-runtime-test'
export type { TestModelConfig } from './resolver/model-runtime-test'
export { TraceEmitter, TracePersister, makeTracePersister, makeNoopTracer, makeTestTracer } from './resolver/tracing'
export type { TraceData } from './resolver/tracing'

// Registry
export { PROVIDERS, getProvider, getProviderIds, populateModels, getModelCost } from './registry'

// Dynamic models (models.dev)
export { initializeModels } from './models-dev'

// Config / persistence
export {
  loadAuth,
  getAuth,
  setAuth,
  removeAuth,
  loadConfig,
  saveConfig,
  setPrimarySelection,
  setBrowserSelection,
} from './config'

// Detection
export { detectProviders, detectDefaultProvider, detectProviderAuthMethods } from './detect'
export type { DetectedProvider, DetectedAuthMethod, ProviderAuthMethodStatus } from './detect'

// ClientRegistry builder
export { buildClientRegistry } from './client-registry-builder'

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

// Local provider config
export { setLocalProviderConfig, getLocalProviderConfig } from './local-config'

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