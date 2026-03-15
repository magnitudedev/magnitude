export const MODEL_DEFAULTS: Record<string, { primary: string; secondary: string; browser: string }> = {
  'anthropic': { primary: 'claude-opus-4-6', secondary: 'claude-sonnet-4-6', browser: 'claude-haiku-4-5' },
  'openai': { primary: 'gpt-5.3-codex', secondary: 'gpt-5.3-codex', browser: 'gpt-5.3-codex' },
  'github-copilot': { primary: 'claude-opus-4.6', secondary: 'claude-sonnet-4.6', browser: 'claude-haiku-4.5' },
  'openrouter': { primary: 'anthropic/claude-opus-4.6', secondary: 'anthropic/claude-sonnet-4.6', browser: 'anthropic/claude-haiku-4.5' },
  'vercel': { primary: 'anthropic/claude-opus-4.6', secondary: 'anthropic/claude-sonnet-4.6', browser: 'anthropic/claude-haiku-4.5' },
  'cerebras': { primary: 'zai-glm-4.7', secondary: 'zai-glm-4.7', browser: 'zai-glm-4.7' },
  'amazon-bedrock': { primary: 'us.anthropic.claude-opus-4-6-v1', secondary: 'us.anthropic.claude-sonnet-4-6-v1', browser: 'us.anthropic.claude-haiku-4-5-v1' },
  'google-vertex-anthropic': { primary: 'claude-opus-4-6@default', secondary: 'claude-sonnet-4-6@default', browser: 'claude-haiku-4-5@default' },
  'google': { primary: 'gemini-3.1-pro-preview', secondary: 'gemini-3-flash-preview', browser: 'gemini-3-flash-preview' },
  'google-vertex': { primary: 'gemini-3.1-pro-preview', secondary: 'gemini-3-flash-preview', browser: 'gemini-3-flash-preview' },
  'minimax': { primary: 'MiniMax-M2.5', secondary: 'MiniMax-M2.5', browser: 'MiniMax-M2.5' },
  'zai': { primary: 'glm-5', secondary: 'glm-5', browser: 'glm-5' },
}

/** OAuth-specific overrides (e.g. OpenAI ChatGPT Plus/Pro gets newer models) */
export const MODEL_OAUTH_DEFAULTS: Record<string, { primary: string; secondary: string; browser: string }> = {
  'openai': { primary: 'gpt-5.3-codex', secondary: 'gpt-5.3-codex-spark', browser: 'gpt-5.3-codex' },
}

/** Get the default primary/secondary/browser model IDs for a provider */
export function getDefaultModels(
  providerId: string,
  isOAuth: boolean,
): { primary: string; secondary: string; browser: string } {
  if (isOAuth && MODEL_OAUTH_DEFAULTS[providerId]) {
    return MODEL_OAUTH_DEFAULTS[providerId]
  }
  return MODEL_DEFAULTS[providerId]!
}