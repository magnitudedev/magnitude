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
    id: 'magnitude',
    name: 'Magnitude',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://app.magnitude.dev/api/v1',
    models: [
      staticModel({ id: 'qwen3.6-plus', name: 'Qwen 3.6 Plus', family: 'qwen', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 200000, contextWindow: 200000 }),
      staticModel({ id: 'glm-4.7', name: 'GLM-4.7', family: 'glm', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 202000, contextWindow: 202000 }),
      staticModel({ id: 'glm-5', name: 'GLM-5', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 202000, contextWindow: 202000 }),
      staticModel({ id: 'glm-5.1', name: 'GLM-5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 202000, contextWindow: 202000 }),
      staticModel({ id: 'kimi-k2.5', name: 'Kimi K2.5', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 262000, contextWindow: 262000 }),
      staticModel({ id: 'minimax-m2.5', name: 'MiniMax M2.5', family: 'minimax', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 196000, contextWindow: 196000 }),
      staticModel({ id: 'minimax-m2.7', name: 'MiniMax M2.7', family: 'minimax', releaseDate: '2025-03-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 196000, contextWindow: 196000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['MAGNITUDE_API_KEY'] },
    ],
    providerFamily: 'cloud',
    inventoryMode: 'static',
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    bamlProvider: 'anthropic',
    models: [
      staticModel({ id: 'claude-opus-4-6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      staticModel({ id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 200000 }),
      staticModel({ id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, maxOutputTokens: 16000, contextWindow: 200000 }),
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
      staticModel({ id: 'gpt-5.4', name: 'GPT-5.4', family: 'gpt', releaseDate: '2026-06-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 400000 }),
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
    id: 'openrouter',
    name: 'OpenRouter',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    models: [
      staticModel({ id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      staticModel({ id: 'anthropic/claude-sonnet-4.6', name: 'Claude Sonnet 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 200000 }),
      staticModel({ id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, maxOutputTokens: 16000, contextWindow: 200000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['OPENROUTER_API_KEY'] },
    ],
  },
  {
    id: 'vercel',
    name: 'Vercel AI Gateway',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://ai-gateway.vercel.sh/v1',
    models: [
      staticModel({ id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      staticModel({ id: 'anthropic/claude-sonnet-4.6', name: 'Claude Sonnet 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 200000 }),
      staticModel({ id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, maxOutputTokens: 16000, contextWindow: 200000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['AI_GATEWAY_API_KEY'] },
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
    name: 'Z.AI',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://api.z.ai/api/paas/v4',
    models: [
      staticModel({ id: 'glm-5', name: 'GLM-5', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'glm-4.7', name: 'GLM-4.7', family: 'glm', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['ZHIPU_API_KEY'] },
    ],
  },
  {
    id: 'zai-coding-plan',
    name: 'Z.AI Coding Plan',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://api.z.ai/api/coding/paas/v4',
    models: [
      staticModel({ id: 'glm-5.1', name: 'GLM-5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'glm-5', name: 'GLM-5', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'glm-4.7', name: 'GLM-4.7', family: 'glm', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['ZHIPU_API_KEY'] },
    ],
  },
  {
    id: 'moonshotai',
    name: 'Moonshot AI',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://api.moonshot.ai/v1',
    models: [
      staticModel({ id: 'kimi-k2.5', name: 'Kimi K2.5', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['MOONSHOT_API_KEY'] },
    ],
  },
  {
    id: 'kimi-for-coding',
    name: 'Kimi for Coding',
    bamlProvider: 'anthropic',
    defaultBaseUrl: 'https://api.kimi.com/coding',
    models: [
      staticModel({ id: 'k2p5', name: 'K2p5', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144 }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['KIMI_API_KEY'] },
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
    id: 'fireworks-ai',
    name: 'Fireworks AI',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'https://api.fireworks.ai/inference/v1',
    models: [
      staticModel({ id: 'accounts/fireworks/routers/kimi-k2p5-turbo', name: 'Kimi K2.5 Turbo (Fire Pass)', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true }),
      staticModel({ id: 'accounts/fireworks/models/glm-5p1', name: 'GLM 5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true }),
    ],
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['FIREWORKS_API_KEY'] },
    ],
    providerFamily: 'cloud',
    inventoryMode: 'dynamic',
  },
  {
    id: 'lmstudio',
    name: 'LM Studio',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'http://localhost:1234/v1',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local endpoint' },
      { type: 'api-key', label: 'Optional API key', envKeys: ['LMSTUDIO_API_KEY'] },
    ],
    providerFamily: 'local',
    inventoryMode: 'dynamic',
    localDiscoveryStrategy: 'openai-models',
  },
  {
    id: 'ollama',
    name: 'Ollama',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'http://localhost:11434/v1',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local endpoint' },
      { type: 'api-key', label: 'Optional API key', envKeys: ['OLLAMA_API_KEY'] },
    ],
    providerFamily: 'local',
    inventoryMode: 'dynamic',
    localDiscoveryStrategy: 'ollama-hybrid',
  },
  {
    id: 'llama.cpp',
    name: 'llama.cpp',
    bamlProvider: 'openai-generic',
    defaultBaseUrl: 'http://localhost:8080',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local endpoint' },
      { type: 'api-key', label: 'Optional API key', envKeys: ['LLAMA_CPP_API_KEY'] },
    ],
    providerFamily: 'local',
    inventoryMode: 'dynamic',
    localDiscoveryStrategy: 'openai-models-best-effort',
  },
  {
    id: 'openai-compatible-local',
    name: 'OpenAI-compatible local',
    bamlProvider: 'openai-generic',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local endpoint' },
      { type: 'api-key', label: 'Optional API key', envKeys: ['OPENAI_COMPAT_LOCAL_API_KEY'] },
    ],
    providerFamily: 'local',
    inventoryMode: 'dynamic',
    localDiscoveryStrategy: 'openai-models-best-effort',
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
  if (!provider) return
  if (provider.inventoryMode === 'dynamic' || provider.models.length > 0) {
    provider.models = models
  }
}

export function getModelCost(providerId: string, modelId: string): { input: number; output: number; cache_read?: number; cache_write?: number } | null {
  const provider = getProvider(providerId)
  if (!provider) return null
  const model = provider.models.find(m => m.id === modelId)
  return model?.cost ?? null
}