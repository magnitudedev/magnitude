/**
 * Provider and model type definitions for multi-provider LLM support.
 */

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
  id: string              // API model ID, e.g. "claude-sonnet-4-5-20250929"
  name: string            // Display name, e.g. "Claude Sonnet 4.5"
  contextWindow?: number
  maxOutputTokens?: number
  supportsToolCalls?: boolean
  supportsReasoning?: boolean
  // Extended fields from models.dev
  cost?: { input: number; output: number; cache_read?: number; cache_write?: number }
  family?: string
  releaseDate?: string
  status?: 'alpha' | 'beta' | 'deprecated'
}

export interface ProviderDefinition {
  id: string                    // e.g. "anthropic", "openrouter", "local"
  name: string                  // Display name: "Anthropic", "OpenRouter", etc.
  bamlProvider: BamlProviderType
  defaultBaseUrl?: string       // For openai-generic providers
  defaultModel?: string         // Sensible default model (API key auth)
  defaultOAuthModel?: string    // Default model for OAuth/subscription auth (if different)
  defaultSecondaryModel?: string // Default secondary (faster) model
  defaultBrowserModel?: string  // Default browser agent model
  models: ModelDefinition[]     // Known models for this provider
  authMethods: AuthMethodDef[]  // All supported auth methods, in display order
  oauthOnlyModelIds?: string[]  // Model IDs that require OAuth (hidden for API key users)
}

// Stored auth info — discriminated union
export type AuthInfo =
  | ApiKeyAuth
  | OAuthAuth
  | AwsAuth
  | GcpAuth

export interface ApiKeyAuth {
  type: 'api'
  key: string
}

export interface OAuthAuth {
  type: 'oauth'
  accessToken: string
  refreshToken: string
  expiresAt: number       // Unix timestamp ms
  accountId?: string      // ChatGPT account ID, GitHub Enterprise URL, etc.
  providerSpecific?: Record<string, any>
}

export interface AwsAuth {
  type: 'aws'
  profile?: string
  region?: string
}

export interface GcpAuth {
  type: 'gcp'
  credentialsPath: string
  project?: string
  location?: string
}

// Persisted config (non-secret preferences)
export interface ModelSelection {
  providerId: string
  modelId: string
}

export interface MagnitudeConfig {
  primaryModel?: ModelSelection | null
  secondaryModel?: ModelSelection | null
  browserModel?: ModelSelection | null
  providerOptions?: Record<string, ProviderOptions>
  setupComplete?: boolean
  machineId?: string
  telemetry?: boolean
  memory?: boolean
}

export interface ProviderOptions {
  baseUrl?: string          // Override base URL
  region?: string           // Bedrock region
  project?: string          // Vertex project
  location?: string         // Vertex location
  [key: string]: any        // Other provider-specific options
}
