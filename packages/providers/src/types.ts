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

import type { ProviderModel } from './model/model'

import type { AuthInfo } from '@magnitudedev/storage'
import type { ProviderProtocol } from './protocol/types'

export interface ResolvedProtocol {
  bamlProvider: BamlProviderType
  protocol: ProviderProtocol
}

export interface ProviderDefinition {
  id: string                    // e.g. "anthropic", "openrouter", "lmstudio". Should match models.dev key.
  name: string                  // Display name: "Anthropic", "OpenRouter", etc.
  defaultBaseUrl?: string       // For openai-generic providers
  models: ProviderModel[]       // Known models for this provider
  authMethods: AuthMethodDef[]  // All supported auth methods, in display order
  oauthOnlyModelIds?: string[]  // Model IDs that require OAuth (hidden for API key users)
  providerFamily?: 'local' | 'cloud'
  inventoryMode?: 'static' | 'dynamic'
  localDiscoveryStrategy?: 'openai-models' | 'ollama-hybrid' | 'openai-models-best-effort'
  resolveProtocol: (auth: AuthInfo | null) => ResolvedProtocol
}
