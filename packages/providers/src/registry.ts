/**
 * Provider registry — defines all supported LLM providers.
 *
 * Each provider has a single fallback model for offline use.
 * Full model lists are populated dynamically from models.dev at startup.
 *
 * Provider IDs match models.dev exactly (1:1).
 */

import type { ProviderDefinition, ModelDefinition } from './types'

export const PROVIDERS: ProviderDefinition[] = [
  // ─── Anthropic ──────────────────────────────────────────────
  {
    id: 'anthropic',
    name: 'Anthropic',
    bamlProvider: 'anthropic',
    defaultModel: 'claude-opus-4-6',
    defaultSecondaryModel: 'claude-sonnet-4-6',
    defaultBrowserModel: 'claude-haiku-4-5',
    models: [
      { id: 'claude-opus-4-6', name: 'Claude Opus 4.6', supportsToolCalls: true, maxOutputTokens: 128000 },
      { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', supportsToolCalls: true, maxOutputTokens: 64000 },
    ],
    authMethods: [
      { type: 'oauth-pkce', label: 'Claude Pro/Max subscription' },
      { type: 'api-key', label: 'API key', envKeys: ['ANTHROPIC_API_KEY'] },
    ],
  },

  // ─── OpenAI ─────────────────────────────────────────────────
  {
    id: 'openai',
    name: 'OpenAI',
    bamlProvider: 'openai',
    defaultModel: 'gpt-5.3-codex',
    defaultOAuthModel: 'gpt-5.3-codex',
    defaultSecondaryModel: 'gpt-5.3-codex',
    defaultBrowserModel: 'gpt-5.3-codex',
    oauthOnlyModelIds: ['gpt-5.3-codex-spark'],
    models: [
      { id: 'gpt-5.3-codex', name: 'GPT-5.3 Codex', supportsToolCalls: true, maxOutputTokens: 128000 },
      { id: 'gpt-5.2-codex', name: 'GPT-5.2 Codex', supportsToolCalls: true, maxOutputTokens: 128000 },
      { id: 'gpt-5.2', name: 'GPT-5.2', supportsToolCalls: true, maxOutputTokens: 128000 },
    ],
    authMethods: [
      { type: 'oauth-browser', label: 'ChatGPT Pro/Plus (browser)' },
      { type: 'oauth-device', label: 'ChatGPT Pro/Plus (headless)' },
      { type: 'api-key', label: 'API key', envKeys: ['OPENAI_API_KEY'] },
    ],
  },

  // ─── GitHub Copilot ─────────────────────────────────────────
  {
    id: 'github-copilot',
    name: 'GitHub Copilot',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://api.githubcopilot.com',
    defaultModel: 'claude-opus-4.6',
    defaultSecondaryModel: 'claude-sonnet-4.6',
    defaultBrowserModel: 'claude-haiku-4.5',
    models: [
      { id: 'claude-opus-4.6', name: 'Claude Opus 4.6', supportsToolCalls: true, maxOutputTokens: 64000 },
    ],
    authMethods: [
      { type: 'oauth-device', label: 'GitHub Copilot' },
    ],
  },

  // ─── Google (Gemini API) ────────────────────────────────────
  {
    id: 'google',
    name: 'Google (Gemini)',
    bamlProvider: 'google-ai',
    defaultModel: 'gemini-3.1-pro-preview',
    models: [
      { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro', supportsToolCalls: true, maxOutputTokens: 65536 },
      { id: 'gemini-3-pro-preview', name: 'Gemini 3 Pro', supportsToolCalls: true, maxOutputTokens: 64000 },
      { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash', supportsToolCalls: true, maxOutputTokens: 65536 },
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['GOOGLE_API_KEY', 'GEMINI_API_KEY'] },
    ],
  },

  // ─── OpenRouter ─────────────────────────────────────────────
  {
    id: 'openrouter',
    name: 'OpenRouter',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    defaultModel: 'anthropic/claude-opus-4.6',
    defaultSecondaryModel: 'anthropic/claude-sonnet-4.6',
    defaultBrowserModel: 'anthropic/claude-haiku-4.5',
    models: [
      { id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', supportsToolCalls: true, maxOutputTokens: 128000 },
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['OPENROUTER_API_KEY'] },
    ],
  },

  // ─── Vercel AI Gateway ──────────────────────────────────────
  {
    id: 'vercel',
    name: 'Vercel AI Gateway',
    bamlProvider: 'openai-generic',
    defaultModel: 'anthropic/claude-opus-4.6',
    defaultSecondaryModel: 'anthropic/claude-sonnet-4.6',
    defaultBrowserModel: 'anthropic/claude-haiku-4.5',
    models: [
      { id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', supportsToolCalls: true, maxOutputTokens: 128000 },
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['VERCEL_API_KEY'] },
    ],
  },

  // ─── Google Vertex AI (Gemini models) ───────────────────────
  {
    id: 'google-vertex',
    name: 'Google Vertex AI',
    bamlProvider: 'vertex-ai',
    defaultModel: 'gemini-3.1-pro-preview',
    models: [
      { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro', supportsToolCalls: true, maxOutputTokens: 65536 },
      { id: 'gemini-3-pro-preview', name: 'Gemini 3 Pro', supportsToolCalls: true, maxOutputTokens: 65536 },
      { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash', supportsToolCalls: true, maxOutputTokens: 65536 },
    ],
    authMethods: [
      { type: 'gcp-credentials', label: 'Service account', envKeys: ['GOOGLE_APPLICATION_CREDENTIALS'] },
    ],
  },

  // ─── Google Vertex AI (Anthropic models) ────────────────────
  {
    id: 'google-vertex-anthropic',
    name: 'Vertex AI (Anthropic)',
    bamlProvider: 'vertex-ai',
    defaultModel: 'claude-opus-4-6@default',
    defaultSecondaryModel: 'claude-sonnet-4-6@default',
    defaultBrowserModel: 'claude-haiku-4-5@default',
    models: [
      { id: 'claude-opus-4-6@default', name: 'Claude Opus 4.6 (Vertex)', supportsToolCalls: true, maxOutputTokens: 128000 },
    ],
    authMethods: [
      { type: 'gcp-credentials', label: 'Service account', envKeys: ['GOOGLE_APPLICATION_CREDENTIALS'] },
    ],
  },

  // ─── Amazon Bedrock ─────────────────────────────────────────
  {
    id: 'amazon-bedrock',
    name: 'Amazon Bedrock',
    bamlProvider: 'aws-bedrock',
    defaultModel: 'us.anthropic.claude-opus-4-6-v1',
    defaultSecondaryModel: 'us.anthropic.claude-sonnet-4-6-v1',
    defaultBrowserModel: 'us.anthropic.claude-haiku-4-5-v1',
    models: [
      { id: 'us.anthropic.claude-opus-4-6-v1', name: 'Claude Opus 4.6 (Bedrock)', supportsToolCalls: true, maxOutputTokens: 128000 },
    ],
    authMethods: [
      { type: 'aws-chain', label: 'AWS credentials', envKeys: ['AWS_ACCESS_KEY_ID', 'AWS_PROFILE', 'AWS_DEFAULT_REGION'] },
    ],
  },

  // ─── MiniMax ──────────────────────────────────────────────
  {
    id: 'minimax',
    name: 'MiniMax',
    bamlProvider: 'anthropic',
    defaultBaseUrl: 'https://api.minimax.io/anthropic',
    defaultModel: 'MiniMax-M2.5',
    defaultSecondaryModel: 'MiniMax-M2.5',
    defaultBrowserModel: 'MiniMax-M2.5',
    models: [
      { id: 'MiniMax-M2.5', name: 'MiniMax M2.5', supportsToolCalls: true, maxOutputTokens: 131072 },
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['MINIMAX_API_KEY'] },
    ],
  },

  // ─── Z.AI (Zhipu AI) ──────────────────────────────────────
  {
    id: 'zai',
    name: 'Z.AI (Zhipu AI)',
    bamlProvider: 'anthropic',
    defaultBaseUrl: 'https://api.z.ai/api/anthropic',
    defaultModel: 'glm-5',
    defaultSecondaryModel: 'glm-5',
    defaultBrowserModel: 'glm-5',
    models: [
      { id: 'glm-5', name: 'GLM-5', supportsToolCalls: true, maxOutputTokens: 131072 },
      { id: 'glm-4.7', name: 'GLM-4.7', supportsToolCalls: true, maxOutputTokens: 131072 },
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['ZHIPU_API_KEY'] },
    ],
  },

  // ─── Cerebras ───────────────────────────────────────────────
  {
    id: 'cerebras',
    name: 'Cerebras',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://api.cerebras.ai/v1',
    defaultModel: 'zai-glm-4.7',
    defaultSecondaryModel: 'zai-glm-4.7',
    defaultBrowserModel: 'zai-glm-4.7',
    models: [
      { id: 'gpt-oss-120b', name: 'GPT-OSS 120B', supportsToolCalls: true, maxOutputTokens: 32768 },
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['CEREBRAS_API_KEY'] },
    ],
  },

  // ─── Local (Ollama, LM Studio, llama.cpp, vLLM, etc.) ──────
  {
    id: 'local',
    name: 'Local',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'http://localhost:1234/v1',
    defaultModel: 'local-model',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local (no auth required)' },
    ],
  },
]

/** Look up a provider by ID */
export function getProvider(providerId: string): ProviderDefinition | undefined {
  return PROVIDERS.find(p => p.id === providerId)
}

/** Get all provider IDs */
export function getProviderIds(): string[] {
  return PROVIDERS.map(p => p.id)
}

/**
 * Populate provider model lists from an external source (models.dev).
 * Mutates PROVIDERS in-place so all downstream consumers see updated models.
 */
export function populateModels(
  getModels: (providerId: string) => ModelDefinition[]
): void {
  for (const provider of PROVIDERS) {
    // Skip local provider — models are local, not on models.dev
    if (provider.id === 'local') continue
    const models = getModels(provider.id)
    if (models.length > 0) {
      provider.models = models
    }
  }
}

/** Look up pricing for a specific model. Returns cost per million tokens or null. */
export function getModelCost(providerId: string, modelId: string): { input: number; output: number; cache_read?: number; cache_write?: number } | null {
  const provider = getProvider(providerId)
  if (!provider) return null
  const model = provider.models.find(m => m.id === modelId)
  return model?.cost ?? null
}
