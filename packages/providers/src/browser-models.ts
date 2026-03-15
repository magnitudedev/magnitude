/**
 * Browser-compatible model registry.
 *
 * Hard-coded master list of models known to support the vision/browser agent.
 * Models are matched by prefix against this list per provider.
 */

import type { ModelSelection } from './types'
import type { DetectedProvider } from './detect'

/**
 * Browser-compatible models keyed by provider ID.
 * Values are model ID prefixes — a model matches if its ID starts with any entry.
 */
export const BROWSER_COMPATIBLE_MODELS: Record<string, string[]> = {
  // ─── Direct providers ─────────────────────────────────────────
  'anthropic': [
    'claude-opus-4-6', 'claude-opus-4-5', 'claude-sonnet-4-6', 'claude-sonnet-4-5', 'claude-haiku-4-5',
  ],
  'openai': [
    'gpt-5.4', 'gpt-5.3-codex', 'gpt-5.2-codex',
  ],

  // ─── OpenRouter ───────────────────────────────────────────────
  'openrouter': [
    // Anthropic
    'anthropic/claude-opus-4.6', 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5',
    // Qwen
    'qwen/qwen3.5-397b-a17b', 'qwen/qwen3.5-122b-a10b', 'qwen/qwen3.5-35b-a3b', 'qwen/qwen3.5-27b', 'qwen/qwen3.5-9b',
    'qwen/qwen3-max-thinking', 'qwen/qwen3.5-plus-02-15', 'qwen/qwen3.5-flash-02-23', 'qwen/qwen3-coder-next',
    // Kimi
    'moonshotai/kimi-k2.5', 'moonshotai/kimi-k2-thinking',
    // DeepSeek
    'deepseek/deepseek-v3.2', 'deepseek/deepseek-v3.2-speciale',
    // MiniMax
    'minimax/minimax-m2.5', 'minimax/minimax-m2.1',
    // GLM
    'z-ai/glm-5', 'z-ai/glm-4.7', 'z-ai/glm-4.6v', 'z-ai/glm-4.7-flash',
    // Grok
    'x-ai/grok-4', 'x-ai/grok-4.1-fast',
    // GPT-OSS
    'openai/gpt-oss-120b', 'openai/gpt-oss-20b',
    // Mistral
    'mistralai/mistral-large-2512', 'mistralai/devstral-2512',
    // Arcee
    'arcee-ai/trinity-large-preview:free',
  ],

  // ─── GitHub Copilot ───────────────────────────────────────────
  'github-copilot': [
    'claude-opus-4.6', 'claude-opus-4.5', 'claude-sonnet-4.6', 'claude-sonnet-4.5', 'claude-haiku-4.5',
    'gpt-5.2-codex',
    'grok-code-fast-1',
  ],

  // ─── Vercel ───────────────────────────────────────────────────
  'vercel': [
    // Anthropic
    'anthropic/claude-opus-4.6', 'anthropic/claude-opus-4.5', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-haiku-4.5',
    // OpenAI
    'openai/gpt-5.3-codex', 'openai/gpt-5.2-codex', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b',
    // Qwen
    'alibaba/qwen3.5-flash', 'alibaba/qwen3.5-plus', 'alibaba/qwen3-max-thinking', 'alibaba/qwen3-coder-next',
    // Kimi
    'moonshotai/kimi-k2.5', 'moonshotai/kimi-k2-thinking',
    // DeepSeek
    'deepseek/deepseek-v3.2-exp', 'deepseek/deepseek-v3.2-thinking',
    // MiniMax
    'minimax/minimax-m2.5', 'minimax/minimax-m2.1',
    // GLM
    'zai/glm-5', 'zai/glm-4.7', 'zai/glm-4.6v',
    // Grok
    'xai/grok-4', 'xai/grok-4.1-fast-non-reasoning',
    // Mistral
    'mistral/mistral-large-3', 'mistral/devstral-2',
    // Arcee
    'arcee-ai/trinity-large-preview',
  ],

  // ─── ZAI (direct) ─────────────────────────────────────────────
  'zai': [
    'glm-5', 'glm-4.7', 'glm-4.6v', 'glm-4.7-flash',
  ],

  // ─── MiniMax (direct) ───────────────────────────────────────
  'minimax': [
    'MiniMax-M2.5', 'MiniMax-M2.1',
  ],

  // ─── Cerebras ─────────────────────────────────────────────────
  'cerebras': [
    'gpt-oss-120b', 'qwen-3-235b-a22b-instruct-2507', 'zai-glm-4.7',
  ],

  // ─── Google (Gemini API) ─────────────────────────────────────
  'google': [
    'gemini-3.1-pro-preview', 'gemini-3-pro-preview', 'gemini-3-flash-preview',
  ],

  // ─── Google Vertex AI (Gemini) ─────────────────────────────────
  'google-vertex': [
    'gemini-3.1-pro-preview', 'gemini-3-pro-preview', 'gemini-3-flash-preview',
  ],

  // ─── Proxy providers (Anthropic models only) ──────────────────
  'google-vertex-anthropic': [
    'claude-opus-4-6', 'claude-opus-4-5', 'claude-sonnet-4-6', 'claude-sonnet-4-5', 'claude-haiku-4-5',
  ],
  'amazon-bedrock': [
    'us.anthropic.claude-opus-4-6', 'us.anthropic.claude-opus-4-5', 'us.anthropic.claude-sonnet-4-6', 'us.anthropic.claude-sonnet-4-5', 'us.anthropic.claude-haiku-4-5',
  ],
}

/** Provider priority for auto-detection (scans all, but prefers these in order) */
const BROWSER_PROVIDER_PRIORITY = [
  'anthropic', 'openai', 'github-copilot', 'openrouter', 'vercel', 'cerebras', 'zai', 'minimax', 'amazon-bedrock', 'google-vertex-anthropic', 'google', 'google-vertex',
]

/** Check if a provider+model combination is browser-compatible */
export function isBrowserCompatible(providerId: string, modelId: string): boolean {
  const models = BROWSER_COMPATIBLE_MODELS[providerId]
  if (!models) return false
  return models.some(prefix => modelId.startsWith(prefix))
}

/** Get all browser-compatible model ID prefixes for a provider */
export function getBrowserCompatibleModels(providerId: string): string[] {
  return BROWSER_COMPATIBLE_MODELS[providerId] ?? []
}

/**
 * Auto-detect the best browser model from all connected providers.
 * If preferredProviderId is given (e.g. the user's primary model provider),
 * tries that provider first before falling back to the static priority list.
 * Returns null if no compatible model is found.
 */
export function detectBrowserModel(
  detectedProviders: DetectedProvider[],
  preferredProviderId?: string | null,
): ModelSelection | null {
  // Build scan order: preferred provider first, then static priority (skipping preferred to avoid double-check)
  const scanOrder = preferredProviderId
    ? [preferredProviderId, ...BROWSER_PROVIDER_PRIORITY.filter(p => p !== preferredProviderId)]
    : BROWSER_PROVIDER_PRIORITY

  for (const pid of scanOrder) {
    const detected = detectedProviders.find(d => d.provider.id === pid)
    if (!detected) continue

    const compatible = getBrowserCompatibleModels(pid)
    for (const prefix of compatible) {
      // Find an actual model in the provider's model list that matches
      const model = detected.provider.models.find(m => m.id.startsWith(prefix))
      if (model) {
        return { providerId: pid, modelId: model.id }
      }
    }
  }
  return null
}
