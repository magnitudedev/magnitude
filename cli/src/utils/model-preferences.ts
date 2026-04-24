import type { MagnitudeSlot } from '@magnitudedev/agent'

const rest = (model: string): Record<MagnitudeSlot, string> => ({
  lead: model,
  worker: model,
})

const tiered = (lead: string, worker: string): Record<MagnitudeSlot, string> => ({
  lead,
  worker,
})

export const MODEL_DEFAULTS: Record<string, Record<MagnitudeSlot, string>> = {
  'anthropic': tiered('claude-opus-4-7', 'claude-sonnet-4-6'),
  'openai': tiered('gpt-5.5', 'gpt-5.5'),
  'openrouter': tiered('z-ai/glm-5.1', 'moonshotai/kimi-k2.6'),
  'vercel': tiered('zai/glm-5.1', 'moonshotai/kimi-k2.6'),
  'cerebras': rest('gpt-oss-120b'),
  'minimax': rest('MiniMax-M2.7'),
  'zai': rest('glm-5.1'),
  'zai-coding-plan': rest('glm-5.1'),
  'moonshotai': rest('kimi-k2.6'),
  'kimi-for-coding': rest('k2p6'),
  'fireworks-ai': rest('accounts/fireworks/models/kimi-k2p6'),
  'magnitude': tiered('glm-5.1', 'kimi-k2.6'),
}

/** OAuth-specific overrides */
export const MODEL_OAUTH_DEFAULTS: Record<string, Record<MagnitudeSlot, string>> = {
  'openai': tiered('gpt-5.5', 'gpt-5.5'),
}

/** Get the default model IDs for all slots for a provider */
export function getDefaultModels(
  providerId: string,
  isOAuth: boolean,
): Record<MagnitudeSlot, string> {
  if (isOAuth && MODEL_OAUTH_DEFAULTS[providerId]) {
    return MODEL_OAUTH_DEFAULTS[providerId]
  }
  return MODEL_DEFAULTS[providerId]!
}
