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
  'anthropic': tiered('claude-opus-4-6', 'claude-sonnet-4-6'),
  'openai': tiered('gpt-5.4', 'gpt-5.3-codex'),
  'openrouter': tiered('anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6'),
  'vercel': tiered('anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6'),
  'cerebras': rest('gpt-oss-120b'),
  'minimax': rest('MiniMax-M2.7'),
  'zai': rest('glm-4.7'),
  'zai-coding-plan': rest('glm-4.7'),
  'moonshotai': rest('kimi-k2.5'),
  'kimi-for-coding': rest('k2p5'),
  'fireworks-ai': rest('accounts/fireworks/routers/kimi-k2p5-turbo'),
  'magnitude': tiered('glm-5.1', 'kimi-k2.5'),
}

/** OAuth-specific overrides */
export const MODEL_OAUTH_DEFAULTS: Record<string, Record<MagnitudeSlot, string>> = {
  'openai': tiered('gpt-5.4', 'gpt-5.3-codex'),
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
