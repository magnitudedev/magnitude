/**
 * Provider and model type definitions for multi-provider LLM support.
 */

export type {
  ModelSelection,
  MagnitudeConfig,
  ProviderOptions,
  ContextLimitPolicy,
  AuthInfo,
  ApiKeyAuth,
  OAuthAuth,
  AwsAuth,
  GcpAuth,
} from '@magnitudedev/storage'

// BAML provider types that map to the BAML runtime
export type BamlProviderType =
  | 'anthropic'
  | 'openai'
  | 'openai-responses'
  | 'openai-generic'
  | 'aws-bedrock'
  | 'vertex-ai'
  | 'google-ai'

// Authentication flow types
export type AuthFlowType =
  | 'api-key'       // Simple API key (env var or manual entry)
  | 'oauth-pkce'    // OAuth PKCE flow (Anthropic Claude Pro/Max)
  | 'oauth-device'  // OAuth device code flow (GitHub Copilot, OpenAI headless)
  | 'oauth-browser' // OAuth browser callback flow (OpenAI browser)
  | 'aws-chain'     // AWS credential chain
  | 'gcp-credentials' // Google Cloud service account
  | 'none'          // No auth needed (local providers)

export interface AuthMethodDef {
  type: AuthFlowType
  label: string           // Display label, e.g. "Claude Pro/Max", "Manual API key"
  envKeys?: string[]      // Env vars to check for this method
}

export interface ModelDefinition {
  id: string
  name: string
  contextWindow: number
  maxContextTokens?: number | null
  supportsToolCalls: boolean
  supportsReasoning: boolean
  cost: { input: number; output: number; cache_read?: number; cache_write?: number }
  family: string
  releaseDate: string
  maxOutputTokens?: number
  supportsVision?: boolean
  description?: string
  status?: 'alpha' | 'beta' | 'deprecated'
  discovery?: {
    primarySource: 'static' | 'models.dev' | 'openrouter-api'
    fetchedAt?: string
  }
}

export interface ProviderDefinition {
  id: string                    // e.g. "anthropic", "openrouter", "lmstudio". Should match models.dev key.
  name: string                  // Display name: "Anthropic", "OpenRouter", etc.
  bamlProvider: BamlProviderType
  defaultBaseUrl?: string       // For openai-generic providers
  models: ModelDefinition[]     // Known models for this provider
  authMethods: AuthMethodDef[]  // All supported auth methods, in display order
  oauthOnlyModelIds?: string[]  // Model IDs that require OAuth (hidden for API key users)
  providerFamily?: 'local' | 'cloud'
  inventoryMode?: 'static' | 'dynamic'
  localDiscoveryStrategy?: 'openai-models' | 'ollama-hybrid' | 'openai-models-best-effort'
}