/**
 * Provider registry — defines all supported LLM providers.
 *
 * Each provider has a static fallback model set for offline use.
 * Dynamic catalog refresh may replace these at runtime.
 */

import type { ProviderDefinition, ModelDefinition } from './types'

const STATIC_MODEL_COST = { input: 0, output: 0 } as const

function staticModel(model: Omit<ModelDefinition, 'contextWindow' | 'supportsReasoning' | 'cost' | 'family' | 'releaseDate' | 'discovery'> & {
  contextWindow?: number
  supportsReasoning?: boolean
  cost?: ModelDefinition['cost']
  family: string
  releaseDate: string
}): ModelDefinition {
  return {
    contextWindow: model.contextWindow ?? 200_000,
    supportsReasoning: model.supportsReasoning ?? false,
    cost: model.cost ?? { ...STATIC_MODEL_COST },
    discovery: { primarySource: 'static' },
    ...model,
  }
}

export const PROVIDERS: ProviderDefinition[] = [
  {
    id: 'anthropic',
    name: 'Anthropic',
    bamlProvider: 'anthropic',
    models: [
      staticModel({ id: 'claude-opus-4-6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      staticModel({ id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 200000 }),
    ],
    authMethods: [
      { type: 'oauth-pkce', label: 'Claude Pro/Max subscription' },
      { type: 'api-key', label: 'API key', envKeys: ['ANTHROPIC_API_KEY'] },
    ],
  },
  {
    id: 'openai',
    name: 'OpenAI',
    bamlProvider: 'openai',
    oauthOnlyModelIds: ['gpt-5.3-codex-spark'],
    models: [
      staticModel({ id: 'gpt-5.3-codex', name: 'GPT-5.3 Codex', family: 'gpt', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 400000 }),
      staticModel({ id: 'gpt-5.2-codex', name: 'GPT-5.2 Codex', family: 'gpt', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 400000 }),
      staticModel({ id: 'gpt-5.2', name: 'GPT-5.2', family: 'gpt', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 400000 }),
    ],
    authMethods: [
      { type: 'oauth-browser', label: 'ChatGPT Pro/Plus (browser)' },
      { type: 'oauth-device', label: 'ChatGPT Pro/Plus (headless)' },
      { type: 'api-key', label: 'API key', envKeys: ['OPENAI_API_KEY'] },
    ],
  },
  {
    id: 'github-copilot',
    name: 'GitHub Copilot',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://api.githubcopilot.com',
    models: [
      staticModel({ id: 'claude-opus-4.6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 200000 }),
    ],
    authMethods: [
      { type: 'oauth-device', label: 'GitHub Copilot' },
    ],
  },
  {
    id: 'google',
    name: 'Google (Gemini)',
    bamlProvider: 'google-ai',
    models: [
      staticModel({ id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro', family: 'gemini', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 65536, contextWindow: 1048576 }),
      staticModel({ id: 'gemini-3-pro-preview', name: 'Gemini 3 Pro', family: 'gemini', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 1048576 }),
      staticModel({ id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash', family: 'gemini', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 65536, contextWindow: 1048576 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['GOOGLE_API_KEY', 'GEMINI_API_KEY'] },
    ],
  },
  {
    id: 'openrouter',
    name: 'OpenRouter',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    models: [
      staticModel({ id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['OPENROUTER_API_KEY'] },
    ],
  },
  {
    id: 'vercel',
    name: 'Vercel AI Gateway',
    bamlProvider: 'openai-generic',
    models: [
      staticModel({ id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['VERCEL_API_KEY'] },
    ],
  },
  {
    id: 'google-vertex',
    name: 'Google Vertex AI',
    bamlProvider: 'vertex-ai',
    models: [
      staticModel({ id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro', family: 'gemini', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 65536, contextWindow: 1048576 }),
      staticModel({ id: 'gemini-3-pro-preview', name: 'Gemini 3 Pro', family: 'gemini', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 65536, contextWindow: 1048576 }),
      staticModel({ id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash', family: 'gemini', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 65536, contextWindow: 1048576 }),
    ],
    authMethods: [
      { type: 'gcp-credentials', label: 'Service account', envKeys: ['GOOGLE_APPLICATION_CREDENTIALS'] },
    ],
  },
  {
    id: 'google-vertex-anthropic',
    name: 'Vertex AI (Anthropic)',
    bamlProvider: 'vertex-ai',
    models: [
      staticModel({ id: 'claude-opus-4-6@default', name: 'Claude Opus 4.6 (Vertex)', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
    ],
    authMethods: [
      { type: 'gcp-credentials', label: 'Service account', envKeys: ['GOOGLE_APPLICATION_CREDENTIALS'] },
    ],
  },
  {
    id: 'amazon-bedrock',
    name: 'Amazon Bedrock',
    bamlProvider: 'aws-bedrock',
    models: [
      staticModel({ id: 'us.anthropic.claude-opus-4-6-v1', name: 'Claude Opus 4.6 (Bedrock)', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
    ],
    authMethods: [
      { type: 'aws-chain', label: 'AWS credentials', envKeys: ['AWS_ACCESS_KEY_ID', 'AWS_PROFILE', 'AWS_DEFAULT_REGION'] },
    ],
  },
  {
    id: 'minimax',
    name: 'MiniMax',
    bamlProvider: 'anthropic',
    defaultBaseUrl: 'https://api.minimax.io/anthropic',
    models: [
      staticModel({ id: 'MiniMax-M2.7', name: 'MiniMax M2.7', family: 'minimax', releaseDate: '2025-03-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'MiniMax-M2.5', name: 'MiniMax M2.5', family: 'minimax', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['MINIMAX_API_KEY'] },
    ],
  },
  {
    id: 'zai',
    name: 'Z.AI (Zhipu AI)',
    bamlProvider: 'anthropic',
    defaultBaseUrl: 'https://api.z.ai/api/anthropic',
    models: [
      staticModel({ id: 'glm-5', name: 'GLM-5', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'glm-4.7', name: 'GLM-4.7', family: 'glm', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['ZHIPU_API_KEY'] },
    ],
  },
  {
    id: 'cerebras',
    name: 'Cerebras',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://api.cerebras.ai/v1',
    models: [
      staticModel({ id: 'gpt-oss-120b', name: 'GPT-OSS 120B', family: 'gpt', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 32768, contextWindow: 131072 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['CEREBRAS_API_KEY'] },
    ],
  },
  {
    id: 'local',
    name: 'Local',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'http://localhost:1234/v1',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local (no auth required)' },
    ],
  },
]

const STATIC_FALLBACK_MODELS: Record<string, readonly ModelDefinition[]> = Object.fromEntries(
  PROVIDERS.map((provider) => [provider.id, [...provider.models]]),
)

const providerOrderIndex = new Map(PROVIDERS.map((p, i) => [p.id, i]))

export function compareProviderOrder(a: string, b: string): number {
  const ai = providerOrderIndex.get(a) ?? Infinity
  const bi = providerOrderIndex.get(b) ?? Infinity
  if (ai === bi) return a.localeCompare(b)
  return ai - bi
}

export function getProvider(providerId: string): ProviderDefinition | undefined {
  return PROVIDERS.find(p => p.id === providerId)
}

export function getProviderIds(): string[] {
  return PROVIDERS.map(p => p.id)
}

export function getStaticProviderModels(providerId: string): readonly ModelDefinition[] {
  return STATIC_FALLBACK_MODELS[providerId] ?? []
}

export function setProviderModels(providerId: string, models: ModelDefinition[]): void {
  const provider = getProvider(providerId)
  if (provider && provider.id !== 'local') {
    provider.models = models
  }
}

export function getModelCost(providerId: string, modelId: string): { input: number; output: number; cache_read?: number; cache_write?: number } | null {
  const provider = getProvider(providerId)
  if (!provider) return null
  const model = provider.models.find(m => m.id === modelId)
  return model?.cost ?? null
}