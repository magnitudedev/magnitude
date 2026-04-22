/**
 * Typed option interfaces and capability definitions for the provider protocol system.
 */

// ---------------------------------------------------------------------------
// Option types (typed structs for each BAML provider family)
// ---------------------------------------------------------------------------

export interface AnthropicOptions {
  model: string
  api_key?: string
  max_tokens: number
  temperature?: number
  stop_sequences?: string[]
  allowed_role_metadata?: string[]
  headers?: Record<string, string>
  base_url?: string
  output_config?: { effort: 'low' | 'medium' | 'high' }
  thinking?: { type: 'disabled' | 'enabled' }
}

export interface OpenAIOptions {
  model: string
  api_key?: string
  temperature?: number
  max_tokens?: number
  stop?: string[]
  reasoning?: { effort: 'low' | 'medium' | 'high' }
  headers?: Record<string, string>
  base_url?: string
  store?: boolean
  instructions?: string
}

export interface OpenAIGenericOptions {
  model: string
  api_key?: string
  base_url?: string
  temperature?: number
  max_tokens?: number
  stop?: string[]
  response_format?: { type: 'grammar'; grammar: string }
  grammar?: string
  stream_options?: { include_usage: boolean }
  reasoning_effort?: 'none' | 'low' | 'medium'
  thinking?: { type: 'disabled' | 'enabled' }
  stream?: boolean
  headers?: Record<string, string>
  logprobs?: boolean
  top_logprobs?: number
}

// ---------------------------------------------------------------------------
// Auth strategy — discriminated union covering all 6 patterns
// ---------------------------------------------------------------------------

export type AuthStrategy =
  | { type: 'api-key'; envKeys: string[] }
  | { type: 'oauth-as-api-key' }
  | { type: 'oauth-anthropic' }
  | { type: 'oauth-openai' }
  | { type: 'none' }
  | { type: 'local-optional'; envKeys: string[] }

// ---------------------------------------------------------------------------
// Capability interfaces — each capability IS its serializer
// ---------------------------------------------------------------------------

export interface OpenAIGenericCapabilities {
  grammar?: (grammar: string) => Partial<OpenAIGenericOptions>
  stopSequences?: (seqs: string[]) => Partial<OpenAIGenericOptions>
  maxTokens?: (maxTokens: number) => Partial<OpenAIGenericOptions>
  reasoningEffort?: (modelId: string) => Partial<OpenAIGenericOptions>
  logprobs?: (enabled: boolean, topK?: number) => Partial<OpenAIGenericOptions>
  staticOptions?: Partial<OpenAIGenericOptions>
}

export interface AnthropicCapabilities {
  stopSequences?: (seqs: string[]) => Partial<AnthropicOptions>
  maxTokens?: (maxTokens: number) => Partial<AnthropicOptions>
  reasoningEffort?: (modelId: string) => Partial<AnthropicOptions>
  staticOptions?: Partial<AnthropicOptions>
}

export interface OpenAICapabilities {
  stopSequences?: (seqs: string[]) => Partial<OpenAIOptions>
  maxTokens?: (maxTokens: number) => Partial<OpenAIOptions>
  reasoningEffort?: (modelId: string) => Partial<OpenAIOptions>
  staticOptions?: Partial<OpenAIOptions>
}

// ---------------------------------------------------------------------------
// Provider protocol variants
// ---------------------------------------------------------------------------

export interface OpenAIGenericProviderProtocol {
  type: 'openai-generic'
  authStrategy: AuthStrategy
  defaultBaseUrl?: string
  capabilities: OpenAIGenericCapabilities
}

export interface AnthropicProviderProtocol {
  type: 'anthropic'
  authStrategy: AuthStrategy
  defaultBaseUrl?: string
  capabilities: AnthropicCapabilities
}

export interface OpenAIProviderProtocol {
  type: 'openai'
  authStrategy: AuthStrategy
  capabilities: OpenAICapabilities
}

export type ProviderProtocol =
  | OpenAIGenericProviderProtocol
  | AnthropicProviderProtocol
  | OpenAIProviderProtocol
