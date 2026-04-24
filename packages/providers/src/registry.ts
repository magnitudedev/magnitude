/**
 * Provider registry — defines all supported LLM providers.
 *
 * Each provider has a static fallback model set for offline use.
 * Dynamic catalog refresh may replace these at runtime.
 */

import type { AuthInfo, BamlProviderType, ProviderDefinition, ModelDefinition } from './types'
import type { ProviderProtocol, OpenAIOptions } from './protocol/types'

const STATIC_MODEL_COST = { input: 0, output: 0 } as const
const DEFAULT_TEMPERATURE = 1.0

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

/** Helper for providers that use a single protocol regardless of auth */
function singleProtocol(bamlProvider: BamlProviderType, protocol: ProviderProtocol) {
  return (_auth: AuthInfo | null) => ({ bamlProvider, protocol })
}

export const PROVIDERS: ProviderDefinition[] = [
  {
    id: 'magnitude',
    name: 'Magnitude',
    defaultBaseUrl: 'https://app.magnitude.dev/api/v1',
    models: [
      staticModel({ id: 'glm-4.7', name: 'GLM-4.7', family: 'glm', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true }),
      staticModel({ id: 'glm-5', name: 'GLM-5', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true }),
      staticModel({ id: 'glm-5.1', name: 'GLM-5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 202000, contextWindow: 202000, supportsGrammar: true }),
      staticModel({ id: 'kimi-k2.5', name: 'Kimi K2.5', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true }),
      staticModel({ id: 'kimi-k2.6', name: 'Kimi K2.6', family: 'kimi', releaseDate: '2026-04-20', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true }),
      staticModel({ id: 'minimax-m2.5', name: 'MiniMax M2.5', family: 'minimax', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 196000, contextWindow: 196000, supportsGrammar: false }),
      staticModel({ id: 'minimax-m2.7', name: 'MiniMax M2.7', family: 'minimax', releaseDate: '2025-03-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 196000, contextWindow: 196000, supportsGrammar: false }),
    ],
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
    models: [
      staticModel({ id: 'claude-opus-4-7', name: 'Claude Opus 4.7', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      
      staticModel({ id: 'claude-opus-4-6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      staticModel({ id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 200000 }),
      staticModel({ id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, maxOutputTokens: 16000, contextWindow: 200000 }),
    ],
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
    models: [
      staticModel({ id: 'gpt-5.5', name: 'GPT-5.5', family: 'gpt', releaseDate: '2026-06-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 400000 }),
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
    models: [
      staticModel({ id: 'anthropic/claude-opus-4.7', name: 'Claude Opus 4.7', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      
      staticModel({ id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      staticModel({ id: 'anthropic/claude-sonnet-4.6', name: 'Claude Sonnet 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 200000 }),
      staticModel({ id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, maxOutputTokens: 16000, contextWindow: 200000 }),
      staticModel({ id: 'z-ai/glm-5.1', name: 'GLM 5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true }),
      staticModel({ id: 'moonshotai/kimi-k2.6', name: 'Kimi K2.6', family: 'kimi', releaseDate: '2026-04-20', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true }),
      staticModel({ id: 'deepseek/deepseek-v4-pro', name: 'DeepSeek V4 Pro', family: 'deepseek', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: false }),
    ],
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
    models: [
      staticModel({ id: 'anthropic/claude-opus-4.7', name: 'Claude Opus 4.7', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      
      staticModel({ id: 'anthropic/claude-opus-4.6', name: 'Claude Opus 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 128000, contextWindow: 200000 }),
      staticModel({ id: 'anthropic/claude-sonnet-4.6', name: 'Claude Sonnet 4.6', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 64000, contextWindow: 200000 }),
      staticModel({ id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5', family: 'claude', releaseDate: '2026-01-01', supportsToolCalls: true, supportsReasoning: false, maxOutputTokens: 16000, contextWindow: 200000 }),
      staticModel({ id: 'zai/glm-5.1', name: 'GLM 5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true }),
      staticModel({ id: 'moonshotai/kimi-k2.6', name: 'Kimi K2.6', family: 'kimi', releaseDate: '2026-04-20', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 262000, contextWindow: 262000, supportsGrammar: true }),
      staticModel({ id: 'deepseek/deepseek-v4-pro', name: 'DeepSeek V4 Pro', family: 'deepseek', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: false }),
    ],
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
    id: 'minimax',
    name: 'MiniMax',
    defaultBaseUrl: 'https://api.minimax.io/anthropic',
    models: [
      staticModel({ id: 'MiniMax-M2.7', name: 'MiniMax M2.7', family: 'minimax', releaseDate: '2025-03-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'MiniMax-M2.5', name: 'MiniMax M2.5', family: 'minimax', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
    ],
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
    models: [
      staticModel({ id: 'glm-5.1', name: 'GLM-5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'glm-5', name: 'GLM-5', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'glm-4.7', name: 'GLM-4.7', family: 'glm', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
    ],
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
    models: [
      staticModel({ id: 'glm-5.1', name: 'GLM-5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'glm-5', name: 'GLM-5', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
      staticModel({ id: 'glm-4.7', name: 'GLM-4.7', family: 'glm', releaseDate: '2024-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 1000000 }),
    ],
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
    models: [
      staticModel({ id: 'kimi-k2.6', name: 'Kimi K2.6', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144 }),
      staticModel({ id: 'kimi-k2.5', name: 'Kimi K2.5', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144 }),
    ],
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
    models: [
      staticModel({ id: 'k2p6', name: 'K2p6', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144 }),
      staticModel({ id: 'k2p5', name: 'K2p5', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144 }),
    ],
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
    models: [
      staticModel({ id: 'gpt-oss-120b', name: 'GPT-OSS 120B', family: 'gpt', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 32768, contextWindow: 131072 }),
    ],
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
    models: [
      staticModel({ id: 'accounts/fireworks/models/kimi-k2p6', name: 'Kimi K2.6', family: 'kimi', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true }),
      staticModel({ id: 'accounts/fireworks/models/glm-5p1', name: 'GLM 5.1', family: 'glm', releaseDate: '2025-01-01', supportsToolCalls: true, supportsReasoning: true, maxOutputTokens: 131072, contextWindow: 262144, supportsGrammar: true }),
    ],
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
