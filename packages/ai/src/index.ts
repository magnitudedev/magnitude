// Model types
export type { ModelId, Model } from "./lib/model/canonical-model"
export type {
  ProviderModel,
  ModelCosts,
  ModelDiscovery,
} from "./lib/model/provider-model"

// Provider
export type { ProviderDefinition } from "./lib/execution/provider-definition"
export type { AuthMethod } from "./lib/auth/types"
export { getProvider, getAllProviders, providers } from "./providers/registry"

export { anthropicProvider } from "./providers/anthropic"
export { cerebrasProvider } from "./providers/cerebras"
export { deepseekProvider } from "./providers/deepseek"
export { fireworksAiProvider } from "./providers/fireworks-ai"
export { kimiForCodingProvider } from "./providers/kimi-for-coding"
export { llamaCppProvider } from "./providers/llama.cpp"
export { lmstudioProvider } from "./providers/lmstudio"
export { magnitudeProvider } from "./providers/magnitude"
export { minimaxProvider } from "./providers/minimax"
export { moonshotAiProvider } from "./providers/moonshotai"
export { openaiProvider } from "./providers/openai"
export { openAiCompatibleLocalProvider } from "./providers/openai-compatible-local"
export { ollamaProvider } from "./providers/ollama"
export { openrouterProvider } from "./providers/openrouter"
export { vercelProvider } from "./providers/vercel"
export { zaiCodingPlanProvider } from "./providers/zai-coding-plan"
export { zaiProvider } from "./providers/zai"

// Auth
export type {
  ResolvedAuth,
  ApiKeyAuth,
  OAuthAuth,
  NoAuth,
} from "./lib/auth/types"
export {
  AuthStorage,
  type StoredAuth,
  type StoredApiKey,
  type StoredOAuth,
} from "./lib/auth/storage"
export { resolveEnvAuth } from "./lib/auth/env"
export { ModelAuth, ModelAuthLive } from "./lib/auth/service"

// Prompt
export type { ToolCallId } from "./lib/prompt/ids"
export type {
  TextPart,
  ImagePart,
  ToolCallPart,
  JsonValue,
  PromptPart,
} from "./lib/prompt/parts"
export type {
  UserMessage,
  AssistantMessage,
  ToolResultMessage,
  Message,
  UserPart,
  ToolResultPart,
  TerminalMessage,
} from "./lib/prompt/messages"
export { Prompt, type PromptShape } from "./lib/prompt/prompt"
export { PromptBuilder } from "./lib/prompt/prompt-builder"

// Tools
export type { ToolDefinition } from "./lib/tools/tool-definition"
export { defineTool } from "./lib/tools/tool-definition"

// Response
export type { ResponseUsage } from "./lib/response/usage"
export type { ResponseStreamEvent } from "./lib/response/events"

// Errors
export {
  NotConfigured,
  AuthFailed,
  RateLimited,
  UsageLimitExceeded,
  ContextLimitExceeded,
  InvalidRequest,
  TransportError,
  ParseError,
  type ModelError,
} from "./lib/errors/model-error"
export { classifyGenericError, type ErrorClassifier } from "./lib/errors/classify"

// Codec + Driver
export type { Codec, EncodeOptions } from "./lib/codec/codec"
export { nativeChatCompletionsCodec } from "./lib/codec/native-chat-completions"
export type { Driver } from "./lib/driver/driver"
export { openAIChatCompletionsDriver } from "./lib/driver/openai-chat-completions"

// Execution
export type { BoundModel } from "./lib/execution/bound-model"
export { bindModel } from "./lib/execution/bind"
export { execute } from "./lib/execution/execute"

// Tracing
export { AiTracer, NoopAiTracer, NoopAiTracerLive } from "./lib/tracing/tracer"

// Catalogue
export { ModelCatalogue, ModelCatalogueLive } from "./catalogues/catalogue"
export { CatalogueCache, type CachedData } from "./catalogues/cache"
export {
  CatalogueConfig,
  type ProviderOptions,
  type DiscoveredModel,
  type DiscoveryStatus,
} from "./catalogues/config"
export type { CatalogueSource, CatalogueError } from "./catalogues/types"
export {
  CatalogueTransportError,
  CatalogueAuthError,
  CatalogueSchemaError,
} from "./catalogues/types"
export { mergeProviderModels } from "./catalogues/merge"
export { staticCatalogueSource } from "./catalogues/static/source"
export { modelsDevCatalogueSource } from "./catalogues/models-dev/source"
export { openRouterCatalogueSource } from "./catalogues/openrouter/source"
export { makeLocalDiscoverySource } from "./catalogues/local-discovery/source"

// Wire types
export type {
  ChatCompletionsRequest,
  ChatCompletionsStreamChunk,
  ChatMessage,
  ChatTool,
} from "./lib/wire/chat-completions"

// Advanced utilities
export * from "./lib/jsonish/types"
export * from "./lib/jsonish/parser"
