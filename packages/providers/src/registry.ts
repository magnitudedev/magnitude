/**
 * Provider registry — defines all supported LLM providers.
 *
 * Each provider has a static fallback model set for offline use.
 * Dynamic catalog refresh may replace these at runtime.
 */

import type { AuthInfo, BamlProviderType, ProviderDefinition } from './types'
import type { ProviderModel, ModelCosts } from './model/model'
import type { ProviderProtocol, OpenAIOptions } from './protocol/types'
import { type ModelId } from './model/canonical-model'
import { MODEL_MANIFEST } from './model/model-manifest'

const STATIC_MODEL_COST: ModelCosts = { inputPerM: 0, outputPerM: 0, cacheReadPerM: null, cacheWritePerM: null }
const DEFAULT_TEMPERATURE = 1.0

/** Try to resolve a provider-specific model ID to a canonical ModelId.
 *  First tries direct match, then strips namespace prefix (e.g. "moonshotai/kimi-k2.6" → "kimi-k2.6").
 */
export function tryResolveCanonicalModelId(providerModelId: string): ModelId | null {
  const canonicalIds = new Set(MODEL_MANIFEST.map(m => m.id))
  if (canonicalIds.has(providerModelId)) return providerModelId
  const stripped = providerModelId.includes('/') ? providerModelId.split('/').pop()! : providerModelId
  if (canonicalIds.has(stripped)) return stripped
  return null
}

type ModelInit = Partial<ProviderModel> & {
  id: string
  name: string
  releaseDate: string
}

function providerModels(
  providerId: string,
  providerName: string,
  models: readonly ModelInit[],
): ProviderModel[] {
  return models.map(m => ({
    providerId,
    providerName,
    modelId: tryResolveCanonicalModelId(m.id),
    contextWindow: 200_000,
    maxContextTokens: null,
    maxOutputTokens: null,
    supportsToolCalls: false,
    supportsReasoning: false,
    supportsVision: false,
    costs: { ...STATIC_MODEL_COST },
    ...m,
  }))
}

/** Helper for providers that use a single protocol regardless of auth */
function singleProtocol(bamlProvider: BamlProviderType, protocol: ProviderProtocol) {
  return (_auth: AuthInfo | null) => ({ bamlProvider, protocol })
}

export const PROVIDERS: ProviderDefinition[] = [
  {
    id: 'magnitude',
    name: 'Magnitude',
    defaultBaseUrl: 'https://app.magnitude.dev/api/v1',
    models: providerModels('magnitude', 'Magnitude', [
      { id: 'glm-4.7', name: 'GLM-4.7', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true },
      { id: 'glm-5', name: 'GLM-5', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true },
      { id: 'glm-5.1', name: 'GLM-5.1', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true },
      { id: 'kimi-k2.5', name: 'Kimi K2.5', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true },
      { id: 'kimi-k2.6', name: 'Kimi K2.6', releaseDate: '2026-04-20', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true },
      { id: 'minimax-m2.5', name: 'MiniMax M2.5', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 196000, contextWindow: 196000, supportsGrammar: false },
      { id: 'minimax-m2.7', name: 'MiniMax M2.7', releaseDate: '2025-03-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 196000, contextWindow: 196000, supportsGrammar: false },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['MAGNITUDE_API_KEY'] },
    ],
    providerFamily: 'cloud',
    inventoryMode: 'static',
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['MAGNITUDE_API_KEY'] },
      defaultBaseUrl: 'https://app.magnitude.dev/api/v1',
      capabilities: {
        grammar: (g) => ({ response_format: { type: 'grammar', grammar: g } }),
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        reasoningEffort: (modelId) => ({
          reasoning_effort: modelId.toLowerCase().includes('minimax') ? 'low' : 'none',
        }),
        staticOptions: { stream_options: { include_usage: true }, stream: true, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    models: providerModels('anthropic', 'Anthropic', [
      { id: 'claude-opus-4-7', name: 'Claude Opus 4.7', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
      { id: 'claude-opus-4-6', name: 'Claude Opus 4.6', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
      { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 64000, contextWindow: 200000 },
      { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, supportsVision: true, maxOutputTokens: 16000, contextWindow: 200000 },
    ]),
    authMethods: [
      { type: 'oauth-pkce', label: 'Claude Pro/Max subscription' },
      { type: 'api-key', label: 'API key', envKeys: ['ANTHROPIC_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('anthropic', {
      type: 'anthropic',
      authStrategy: { type: 'oauth-anthropic' },
      capabilities: {
        stopSequences: (seqs) => ({ stop_sequences: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        reasoningEffort: (modelId) => {
          const id = modelId.toLowerCase()
          if (id.includes('opus-4-6') || id.includes('opus-4.6')) return { output_config: { effort: 'low' } }
          if (id.includes('sonnet-4-6') || id.includes('sonnet-4.6')) return { output_config: { effort: 'low' } }
          if ((id.includes('opus-4-5') || id.includes('opus-4.5')) && !id.includes('sonnet')) return { output_config: { effort: 'low' } }
          return {}
        },
        staticOptions: {
          temperature: DEFAULT_TEMPERATURE,
          allowed_role_metadata: ['cache_control'],
        },
      },
    }),
  },
  {
    id: 'openai',
    name: 'OpenAI',
    oauthOnlyModelIds: ['gpt-5.5-codex-spark'],
    models: providerModels('openai', 'OpenAI', [
      { id: 'gpt-5.5', name: 'GPT-5.5', releaseDate: '2026-06-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
      { id: 'gpt-5.4', name: 'GPT-5.4', releaseDate: '2026-06-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
      { id: 'gpt-5.3-codex', name: 'GPT-5.3 Codex', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
      { id: 'gpt-5.2-codex', name: 'GPT-5.2 Codex', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
      { id: 'gpt-5.2', name: 'GPT-5.2', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 400000 },
    ]),
    authMethods: [
      { type: 'oauth-browser', label: 'ChatGPT Pro/Plus (browser)' },
      { type: 'oauth-device', label: 'ChatGPT Pro/Plus (headless)' },
      { type: 'api-key', label: 'API key', envKeys: ['OPENAI_API_KEY'] },
    ],
    resolveProtocol: (auth) => {
      const reasoningEffort: (modelId: string) => Partial<OpenAIOptions> = (modelId) => {
        const id = modelId.toLowerCase()
        if (id.includes('o3') || id.includes('o4') || id.includes('codex') ||
            (id.includes('gpt-5') && !id.includes('5.'))) {
          return { reasoning: { effort: 'low' } }
        }
        return {}
      }

      if (auth?.type === 'oauth' || auth?.type === 'api') {
        // Responses API — no stop sequences, no max_output_tokens
        return {
          bamlProvider: 'openai-responses',
          protocol: {
            type: 'openai',
            authStrategy: { type: 'oauth-openai' },
            capabilities: {
              reasoningEffort,
              // store: false only for oauth — api-key users get API default
              ...(auth?.type === 'oauth' ? { staticOptions: { store: false } } : {}),
              // max_tokens NOT supported for responses API
            },
          },
        }
      }
      // Chat Completions API — supports stop sequences and max_tokens
      return {
        bamlProvider: 'openai',
        protocol: {
          type: 'openai',
          authStrategy: { type: 'oauth-openai' },
          capabilities: {
            stopSequences: (seqs) => ({ stop: seqs }),
            maxTokens: (n) => ({ max_tokens: n }),
            reasoningEffort,
          },
        },
      }
    },
  },
  {
    id: 'openrouter',
    name: 'OpenRouter',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    models: providerModels('openrouter', 'OpenRouter', [
      { id: 'anthropic/claude-opus-4.7', name: 'Claude Opus 4.7', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
      { id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
      { id: 'anthropic/claude-sonnet-4.6', name: 'Claude Sonnet 4.6', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 64000, contextWindow: 200000 },
      { id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, supportsVision: true, maxOutputTokens: 16000, contextWindow: 200000 },
      { id: 'z-ai/glm-5.1', name: 'GLM 5.1', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true },
      { id: 'moonshotai/kimi-k2.6', name: 'Kimi K2.6', releaseDate: '2026-04-20', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true },
      { id: 'deepseek/deepseek-v4-pro', name: 'DeepSeek V4 Pro', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: false },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['OPENROUTER_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['OPENROUTER_API_KEY'] },
      defaultBaseUrl: 'https://openrouter.ai/api/v1',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'vercel',
    name: 'Vercel AI Gateway',
    defaultBaseUrl: 'https://ai-gateway.vercel.sh/v1',
    models: providerModels('vercel', 'Vercel AI Gateway', [
      { id: 'anthropic/claude-opus-4.7', name: 'Claude Opus 4.7', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
      { id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 128000, contextWindow: 200000 },
      { id: 'anthropic/claude-sonnet-4.6', name: 'Claude Sonnet 4.6', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 64000, contextWindow: 200000 },
      { id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, supportsVision: true, maxOutputTokens: 16000, contextWindow: 200000 },
      { id: 'zai/glm-5.1', name: 'GLM 5.1', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true },
      { id: 'moonshotai/kimi-k2.6', name: 'Kimi K2.6', releaseDate: '2026-04-20', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true },
      { id: 'deepseek/deepseek-v4-pro', name: 'DeepSeek V4 Pro', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: false },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['AI_GATEWAY_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['AI_GATEWAY_API_KEY'] },
      defaultBaseUrl: 'https://ai-gateway.vercel.sh/v1',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    defaultBaseUrl: 'https://api.deepseek.com/v1',
    models: providerModels('deepseek', 'DeepSeek', [
      {
        id: 'deepseek-v4-flash',
        name: 'DeepSeek V4 Flash',
        releaseDate: '2026-04-24',
        supportsToolCalls: true,
        supportsReasoning: true,
        maxOutputTokens: 384000,
        contextWindow: 1000000,
        supportsGrammar: false,
      },
      {
        id: 'deepseek-v4-pro',
        name: 'DeepSeek V4 Pro',
        releaseDate: '2026-04-24',
        supportsToolCalls: true,
        supportsReasoning: true,
        maxOutputTokens: 384000,
        contextWindow: 1000000,
        supportsGrammar: false,
      },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['DEEPSEEK_API_KEY'] },
    ],
    providerFamily: 'cloud',
    inventoryMode: 'static',
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['DEEPSEEK_API_KEY'] },
      defaultBaseUrl: 'https://api.deepseek.com/v1',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        reasoningEffort: (_modelId) => ({ thinking: { type: 'disabled' } }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'minimax',
    name: 'MiniMax',
    defaultBaseUrl: 'https://api.minimax.io/anthropic',
    models: providerModels('minimax', 'MiniMax', [
      { id: 'MiniMax-M2.7', name: 'MiniMax M2.7', releaseDate: '2025-03-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 },
      { id: 'MiniMax-M2.5', name: 'MiniMax M2.5', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['MINIMAX_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('anthropic', {
      type: 'anthropic',
      authStrategy: { type: 'api-key', envKeys: ['MINIMAX_API_KEY'] },
      defaultBaseUrl: 'https://api.minimax.io/anthropic',
      capabilities: {
        stopSequences: (seqs) => ({ stop_sequences: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        staticOptions: { temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'zai',
    name: 'Z.AI',
    defaultBaseUrl: 'https://api.z.ai/api/paas/v4',
    models: providerModels('zai', 'Z.AI', [
      { id: 'glm-5.1', name: 'GLM-5.1', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
      { id: 'glm-5', name: 'GLM-5', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
      { id: 'glm-4.7', name: 'GLM-4.7', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['ZHIPU_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['ZHIPU_API_KEY'] },
      defaultBaseUrl: 'https://api.z.ai/api/paas/v4',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        reasoningEffort: (_modelId) => ({ thinking: { type: 'disabled' } }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'zai-coding-plan',
    name: 'Z.AI Coding Plan',
    defaultBaseUrl: 'https://api.z.ai/api/coding/paas/v4',
    models: providerModels('zai-coding-plan', 'Z.AI Coding Plan', [
      { id: 'glm-5.1', name: 'GLM-5.1', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
      { id: 'glm-5', name: 'GLM-5', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
      { id: 'glm-4.7', name: 'GLM-4.7', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 1000000 },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['ZHIPU_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['ZHIPU_API_KEY'] },
      defaultBaseUrl: 'https://api.z.ai/api/coding/paas/v4',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        reasoningEffort: (_modelId) => ({ thinking: { type: 'disabled' } }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'moonshotai',
    name: 'Moonshot AI',
    defaultBaseUrl: 'https://api.moonshot.ai/v1',
    models: providerModels('moonshotai', 'Moonshot AI', [
      { id: 'kimi-k2.6', name: 'Kimi K2.6', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144 },
      { id: 'kimi-k2.5', name: 'Kimi K2.5', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144 },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['MOONSHOT_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['MOONSHOT_API_KEY'] },
      defaultBaseUrl: 'https://api.moonshot.ai/v1',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        reasoningEffort: (_modelId) => ({ thinking: { type: 'disabled' } }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'kimi-for-coding',
    name: 'Kimi for Coding',
    defaultBaseUrl: 'https://api.kimi.com/coding',
    models: providerModels('kimi-for-coding', 'Kimi for Coding', [
      { id: 'k2p6', name: 'K2p6', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144 },
      { id: 'k2p5', name: 'K2p5', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144 },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['KIMI_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('anthropic', {
      type: 'anthropic',
      authStrategy: { type: 'api-key', envKeys: ['KIMI_API_KEY'] },
      defaultBaseUrl: 'https://api.kimi.com/coding',
      capabilities: {
        stopSequences: (seqs) => ({ stop_sequences: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        reasoningEffort: (_modelId) => ({ thinking: { type: 'disabled' } }),
        staticOptions: { temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'cerebras',
    name: 'Cerebras',
    defaultBaseUrl: 'https://api.cerebras.ai/v1',
    models: providerModels('cerebras', 'Cerebras', [
      { id: 'gpt-oss-120b', name: 'GPT-OSS 120B', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 32768, contextWindow: 131072 },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['CEREBRAS_API_KEY'] },
    ],
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['CEREBRAS_API_KEY'] },
      defaultBaseUrl: 'https://api.cerebras.ai/v1',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'fireworks-ai',
    name: 'Fireworks AI',
    defaultBaseUrl: 'https://api.fireworks.ai/inference/v1',
    models: providerModels('fireworks-ai', 'Fireworks AI', [
      { id: 'accounts/fireworks/models/kimi-k2p6', name: 'Kimi K2.6', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true, paradigm: 'native' as const, modelId: 'kimi-k2.6' as const },
      { id: 'accounts/fireworks/models/glm-5p1', name: 'GLM 5.1', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, supportsVision: false, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true },
    ]),
    authMethods: [
      { type: 'api-key', label: 'API key', envKeys: ['FIREWORKS_API_KEY'] },
    ],
    providerFamily: 'cloud',
    inventoryMode: 'dynamic',
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'api-key', envKeys: ['FIREWORKS_API_KEY'] },
      defaultBaseUrl: 'https://api.fireworks.ai/inference/v1',
      capabilities: {
        grammar: (g) => ({ response_format: { type: 'grammar', grammar: g } }),
        stopSequences: (seqs) => ({ stop: seqs }),
        reasoningEffort: (modelId) => ({
          reasoning_effort: modelId.toLowerCase().includes('minimax') ? 'low' : 'none',
        }),
        logprobs: (enabled, topK = 5) => enabled ? { logprobs: true, top_logprobs: topK } : {},
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'lmstudio',
    name: 'LM Studio',
    defaultBaseUrl: 'http://localhost:1234/v1',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local endpoint' },
      { type: 'api-key', label: 'Optional API key', envKeys: ['LMSTUDIO_API_KEY'] },
    ],
    providerFamily: 'local',
    inventoryMode: 'dynamic',
    localDiscoveryStrategy: 'openai-models',
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'local-optional', envKeys: ['LMSTUDIO_API_KEY'] },
      defaultBaseUrl: 'http://localhost:1234/v1',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'ollama',
    name: 'Ollama',
    defaultBaseUrl: 'http://localhost:11434/v1',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local endpoint' },
      { type: 'api-key', label: 'Optional API key', envKeys: ['OLLAMA_API_KEY'] },
    ],
    providerFamily: 'local',
    inventoryMode: 'dynamic',
    localDiscoveryStrategy: 'ollama-hybrid',
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'local-optional', envKeys: ['OLLAMA_API_KEY'] },
      defaultBaseUrl: 'http://localhost:11434/v1',
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'llama.cpp',
    name: 'llama.cpp',
    defaultBaseUrl: 'http://localhost:8080',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local endpoint' },
      { type: 'api-key', label: 'Optional API key', envKeys: ['LLAMA_CPP_API_KEY'] },
    ],
    providerFamily: 'local',
    inventoryMode: 'dynamic',
    localDiscoveryStrategy: 'openai-models-best-effort',
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'local-optional', envKeys: ['LLAMA_CPP_API_KEY'] },
      defaultBaseUrl: 'http://localhost:8080',
      capabilities: {
        grammar: (g) => ({ grammar: g }),
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        logprobs: (enabled, topK = 5) => enabled ? { logprobs: true, top_logprobs: topK } : {},
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
  {
    id: 'openai-compatible-local',
    name: 'OpenAI-compatible local',
    models: [],
    authMethods: [
      { type: 'none', label: 'Local endpoint' },
      { type: 'api-key', label: 'Optional API key', envKeys: ['OPENAI_COMPAT_LOCAL_API_KEY'] },
    ],
    providerFamily: 'local',
    inventoryMode: 'dynamic',
    localDiscoveryStrategy: 'openai-models-best-effort',
    resolveProtocol: singleProtocol('openai-generic', {
      type: 'openai-generic',
      authStrategy: { type: 'local-optional', envKeys: ['OPENAI_COMPAT_LOCAL_API_KEY'] },
      capabilities: {
        stopSequences: (seqs) => ({ stop: seqs }),
        maxTokens: (n) => ({ max_tokens: n }),
        staticOptions: { stream_options: { include_usage: true }, temperature: DEFAULT_TEMPERATURE },
      },
    }),
  },
]

const STATIC_FALLBACK_MODELS: Record<string, readonly ProviderModel[]> = Object.fromEntries(
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

export function getStaticProviderModels(providerId: string): readonly ProviderModel[] {
  return STATIC_FALLBACK_MODELS[providerId] ?? []
}

export function setProviderModels(providerId: string, models: ProviderModel[]): void {
  const provider = getProvider(providerId)
  if (!provider) return
  if (provider.inventoryMode === 'dynamic' || provider.models.length > 0) {
    provider.models = models
  }
}

export function getModelCost(providerId: string, modelId: string): ModelCosts | null {
  const provider = getProvider(providerId)
  if (!provider) return null
  const model = provider.models.find(m => m.id === modelId)
  return model?.costs ?? null
}
