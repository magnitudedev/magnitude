import type { MagnitudeSlot } from '@magnitudedev/agent'

const rest = (model: string): Record<MagnitudeSlot, string> => ({
  lead: model,
  explorer: model,
  planner: model,
  builder: model,
  reviewer: model,
  debugger: model,
  browser: model,
})

const tiered = (lead: string, sub: string, browser: string): Record<MagnitudeSlot, string> => ({
  lead,
  explorer: sub,
  planner: sub,
  builder: sub,
  reviewer: sub,
  debugger: sub,
  browser,
})

export const MODEL_DEFAULTS: Record<string, Record<MagnitudeSlot, string>> = {
  'anthropic': tiered('claude-opus-4-6', 'claude-sonnet-4-6', 'claude-haiku-4-5'),
  'openai': tiered('gpt-5.4', 'gpt-5.3-codex', 'gpt-5.3-codex'),
  'github-copilot': tiered('claude-opus-4.6', 'claude-sonnet-4.6', 'claude-haiku-4.5'),
  'openrouter': tiered('anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-haiku-4.5'),
  'vercel': tiered('anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-haiku-4.5'),
  'amazon-bedrock': tiered('us.anthropic.claude-opus-4-6-v1', 'us.anthropic.claude-sonnet-4-6-v1', 'us.anthropic.claude-haiku-4-5-v1'),
  'google-vertex-anthropic': tiered('claude-opus-4-6@default', 'claude-sonnet-4-6@default', 'claude-haiku-4-5@default'),
  'google': tiered('gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-3-flash-preview'),
  'google-vertex': tiered('gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-3-flash-preview'),
  'cerebras': rest('gpt-oss-120b'),
  'minimax': rest('MiniMax-M2.7'),
  'zai': tiered('glm-5', 'glm-4.7', 'glm-4.7-flash'),
  'zai-coding-plan': tiered('glm-5.1', 'glm-4.7', 'glm-4.7-flash'),
}

/** OAuth-specific overrides */
export const MODEL_OAUTH_DEFAULTS: Record<string, Record<MagnitudeSlot, string>> = {
  'openai': tiered('gpt-5.4', 'gpt-5.3-codex', 'gpt-5.3-codex'),
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
