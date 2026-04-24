/**
 * Browser-compatible model registry.
 *
 * Hard-coded master list of models known to support the vision/browser agent.
 * Models are matched by prefix against this list per provider.
 */

import type { ModelSelection } from './types'

const LOCAL_PROVIDER_IDS = new Set(['lmstudio', 'ollama', 'llama.cpp', 'openai-compatible-local'])

const LOCAL_COMPATIBLE_FAMILY_PATTERNS: RegExp[] = [
  /qwen/,
  /gpt[-_. ]?oss/,
  /gpt[-_. ]?\d/,
  /claude/,
  /gemini/,
  /gemma/,
  /glm/,
  /kimi/,
  /minimax/,
  /grok/,
  /deepseek/,
  /mistral/,
]

function isLocalProvider(providerId: string): boolean {
  return LOCAL_PROVIDER_IDS.has(providerId)
}

function normalizeLocalModelId(modelId: string): string {
  let value = modelId.trim().toLowerCase()

  // trim query/hash fragments occasionally present in copied IDs
  value = value.split('?')[0]?.split('#')[0] ?? value

  // strip common file suffixes
  value = value.replace(/\.gguf$/g, '')

  // strip ollama/local tag suffixes after ":" while keeping root model name
  // e.g. qwen3:latest -> qwen3, gpt-oss:120b -> gpt-oss
  value = value.replace(/:([a-z0-9._-]+)$/g, '')

  // remove common local quant/build tokens
  value = value
    .replace(/[-_.](q\d+(_k_[a-z0-9]+)?|iq\d+|fp16|fp8|f16|f32)\b/g, '')
    .replace(/[-_.](instruct|chat|preview|latest)\b/g, '')

  return value
}

function collectKnownCompatiblePrefixes(): string[] {
  const all = Object.values(BROWSER_COMPATIBLE_MODELS).flat()
  const unique = new Set(all.map((entry) => entry.toLowerCase()))
  return Array.from(unique)
}

function extractCanonicalFamilyCandidates(modelId: string): string[] {
  const original = modelId.trim().toLowerCase()
  const normalized = normalizeLocalModelId(original)
  const bare = normalized.split('/').at(-1) ?? normalized
  const noSeparators = bare.replace(/[-_.]/g, '')
  return Array.from(new Set([original, normalized, bare, noSeparators]))
}

function matchesKnownCompatibleFamily(candidates: string[]): boolean {
  const knownCompatiblePrefixes = collectKnownCompatiblePrefixes()

  return candidates.some((candidate) => {
    if (knownCompatiblePrefixes.some((prefix) => candidate.startsWith(prefix) || prefix.includes(candidate))) {
      return true
    }
    return LOCAL_COMPATIBLE_FAMILY_PATTERNS.some((pattern) => pattern.test(candidate))
  })
}

/**
 * Browser-compatible models keyed by provider ID.
 * Values are model ID prefixes — a model matches if its ID starts with any entry.
 */
export const BROWSER_COMPATIBLE_MODELS: Record<string, string[]> = {
  // ─── Direct providers ─────────────────────────────────────────
  'anthropic': [
    'claude-opus-4-7', 'claude-opus-4-6', 'claude-opus-4-5', 'claude-sonnet-4-6', 'claude-sonnet-4-5', 'claude-haiku-4-5',
  ],
  'openai': [
    'gpt-5.5', 'gpt-5.4', 'gpt-5.3-codex', 'gpt-5.2-codex',
  ],

  // ─── OpenRouter ───────────────────────────────────────────────
  'openrouter': [
    // Anthropic
    'anthropic/claude-opus-4.7', 'anthropic/claude-opus-4.6', 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5',
    // Qwen
    'qwen/qwen3.5-397b-a17b', 'qwen/qwen3.5-122b-a10b', 'qwen/qwen3.5-35b-a3b', 'qwen/qwen3.5-27b', 'qwen/qwen3.5-9b',
    'qwen/qwen3-max-thinking', 'qwen/qwen3.5-plus-02-15', 'qwen/qwen3.5-flash-02-23', 'qwen/qwen3-coder-next',
    // Kimi
    'moonshotai/kimi-k2.6', 'moonshotai/kimi-k2.5', 'moonshotai/kimi-k2-thinking',
    // DeepSeek
    'deepseek/deepseek-v3.2', 'deepseek/deepseek-v3.2-speciale',
    // MiniMax
    'minimax/minimax-m2.7', 'minimax/minimax-m2.5', 'minimax/minimax-m2.1',
    // GLM
    'z-ai/glm-5.1', 'z-ai/glm-5', 'z-ai/glm-4.7', 'z-ai/glm-4.6v',
    // DeepSeek
    'deepseek/deepseek-v4-pro',
    // Grok
    'x-ai/grok-4', 'x-ai/grok-4.1-fast',
    // GPT-OSS
    'openai/gpt-oss-120b', 'openai/gpt-oss-20b',
    // Mistral
    'mistralai/mistral-large-2512', 'mistralai/devstral-2512',
    // Arcee
    'arcee-ai/trinity-large-preview:free',
  ],

  // ─── Vercel ───────────────────────────────────────────────────
  'vercel': [
    // Anthropic
    'anthropic/claude-opus-4.7', 'anthropic/claude-opus-4.6', 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5',
    // OpenAI
    'openai/gpt-5.5', 'openai/gpt-5.3-codex', 'openai/gpt-5.2-codex', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b',
    // Qwen
    'alibaba/qwen3.5-flash', 'alibaba/qwen3.5-plus', 'alibaba/qwen3-max-thinking', 'alibaba/qwen3-coder-next',
    // Kimi
    'moonshotai/kimi-k2.6', 'moonshotai/kimi-k2.5', 'moonshotai/kimi-k2-thinking',
    // DeepSeek
    'deepseek/deepseek-v3.2-exp', 'deepseek/deepseek-v3.2-thinking',
    // MiniMax
    'minimax/minimax-m2.7', 'minimax/minimax-m2.5', 'minimax/minimax-m2.1',
    // GLM
    'zai/glm-5.1', 'zai/glm-5', 'zai/glm-4.7', 'zai/glm-4.6v',
    // DeepSeek
    'deepseek/deepseek-v4-pro',
    // Grok
    'xai/grok-4', 'xai/grok-4.1-fast-non-reasoning',
    // Mistral
    'mistral/mistral-large-3', 'mistral/devstral-2',
    // Arcee
    'arcee-ai/trinity-large-preview',
  ],

  // ─── ZAI (direct) ─────────────────────────────────────────────
  'zai': [
    'glm-5.1', 'glm-5', 'glm-4.7', 'glm-4.6v',
  ],

  // ─── MiniMax (direct) ───────────────────────────────────────
  'minimax': [
    'MiniMax-M2.7', 'MiniMax-M2.5', 'MiniMax-M2.1',
  ],

  // ─── Cerebras ─────────────────────────────────────────────────
  'cerebras': [
    'gpt-oss-120b', 'qwen-3-235b-a22b-instruct-2507', 'zai-glm-5.1', 'zai-glm-4.7',
  ],

}

/** Check if a provider+model combination is browser-compatible */
export function isBrowserCompatible(providerId: string, modelId: string): boolean {
  const models = BROWSER_COMPATIBLE_MODELS[providerId]
  if (models?.some(prefix => modelId.startsWith(prefix))) return true

  if (!isLocalProvider(providerId)) {
    return false
  }

  const candidates = extractCanonicalFamilyCandidates(modelId)
  return matchesKnownCompatibleFamily(candidates)
}

/** Get all browser-compatible model ID prefixes for a provider */
export function getBrowserCompatibleModels(providerId: string): string[] {
  return BROWSER_COMPATIBLE_MODELS[providerId] ?? []
}

