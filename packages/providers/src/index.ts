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

// Model slot types
export type { ModelSlot, ResolvedModel, CallUsage, SlotUsage } from './provider-state'

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

// Constants

// Reasoning effort
export { getLowestEffortOptions } from './reasoning-effort'

// Provider state (runtime singleton)
export {
  resolveModel,
  getPrimaryProviderId,
  getPrimaryModelId,
  getClientRegistry,
  isAnthropicOAuth,
  isOpenAICodex,
  isCopilotCodex,
  getCodexAuth,
  getCopilotCodexAuth,
  setModel,
  setPrimaryModel,
  setSecondaryModel,
  setBrowserModel,
  clearPrimaryModel,
  clearSecondaryModel,
  clearBrowserModel,
  initializeProviderState,
  getProviderSummary,
  ensureValidAuth,
  getPrimaryModelContextWindow,
  getModelContextWindow,
  validateModelSwitch,
  getSlotUsage,
  resetSlotUsage,
} from './provider-state'

// Provider client
export { createProviderClient } from './provider-client'
export type { ProviderClient } from './provider-client'

// Usage calculation
export { buildUsage, calculateCosts } from './usage'

// Output normalization
export { normalizeModelOutput, normalizeQuotesInString } from './output-normalization'

// Model proxy
export { createModelProxy, primary, secondary, browser, onTrace } from './model-proxy'
export type { ModelProxy, ChatStream, CollectorData, TraceData } from './model-proxy'


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
