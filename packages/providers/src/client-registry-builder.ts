/**
 * Builds a BAML ClientRegistry for runtime provider/model switching.
 *
 * The ClientRegistry overrides the statically-defined BAML client so that
 * CodingAgentChat (which uses `ChatClientNoRetryAnthropicOnly`) can be
 * pointed at any provider/model at runtime.
 */

import { ClientRegistry } from '@magnitudedev/llm-core'
import { logger } from '@magnitudedev/logger'
import { getProvider } from './registry'
import { loadConfig } from './config'
import { getLocalProviderConfig } from './local-config'
import { getLowestEffortOptions } from './reasoning-effort'
import type { AuthInfo, ProviderDefinition, ProviderOptions } from './types'

/** The BAML client name used by CodingAgentChat */
const CHAT_CLIENT_NAME = 'ChatClientNoRetryAnthropicOnly'

/**
 * Build a ClientRegistry targeting the given provider + model.
 *
 * Returns undefined if no valid auth is available (falls back to BAML static client).
 */
export function buildClientRegistry(
  providerId: string,
  modelId: string,
  auth: AuthInfo | null,
  stopSequences?: string[],
): ClientRegistry | undefined {
  const def = getProvider(providerId)
  if (!def) {
    logger.warn(`[Provider] Unknown provider: ${providerId}`)
    return undefined
  }

  const config = loadConfig()
  const providerOpts = config.providerOptions?.[providerId]

  // For OpenAI with OAuth (ChatGPT subscription) or Copilot with Codex models,
  // use openai-responses provider which hits the /responses endpoint
  const bamlProvider = (providerId === 'openai' && (auth?.type === 'oauth' || auth?.type === 'api'))
    ? 'openai-responses' as const
    : (providerId === 'github-copilot' && modelId.includes('codex'))
      ? 'openai-responses' as const
      : def.bamlProvider

  const modelDef = def.models.find(m => m.id === modelId)
  const maxOutputTokens = modelDef?.maxOutputTokens

  let options = buildOptions({ ...def, bamlProvider }, modelId, auth, providerOpts, stopSequences, maxOutputTokens)
  if (!options) return undefined

  // Apply lowest reasoning effort for models that support it.
  // Magnitude provides its own think tool, so built-in model thinking is redundant.
  const effort = getLowestEffortOptions(providerId, modelId, bamlProvider)
  if (effort) {
    options = deepMerge(options, effort.optionsMerge)
    logger.info(`[Provider] Applied low reasoning effort: ${effort.label}`)
  }

  const cr = new ClientRegistry()
  cr.addLlmClient(CHAT_CLIENT_NAME, bamlProvider, options)
  cr.setPrimary(CHAT_CLIENT_NAME)
  return cr
}

// ---------------------------------------------------------------------------
// Option builders per BAML provider type
// ---------------------------------------------------------------------------

function buildOptions(
  def: ProviderDefinition,
  modelId: string,
  auth: AuthInfo | null,
  providerOpts?: ProviderOptions,
  stopSequences?: string[],
  maxOutputTokens?: number,
): Record<string, any> | undefined {
  switch (def.bamlProvider) {
    case 'anthropic':
      if (def.defaultBaseUrl) {
        return buildAnthropicCompatibleOptions(def, modelId, auth, providerOpts, stopSequences, maxOutputTokens)
      }
      return buildAnthropicOptions(modelId, auth, stopSequences, maxOutputTokens)
    case 'openai':
    case 'openai':
      return buildOpenAIOptions(modelId, auth, stopSequences, maxOutputTokens)
    case 'openai-responses':
      if (def.id === 'github-copilot') {
        return buildCopilotCodexOptions(modelId, auth, stopSequences, maxOutputTokens)
      }
      return buildOpenAIResponsesOptions(modelId, auth, stopSequences, maxOutputTokens)
    case 'openai-generic':
      return buildOpenAIGenericOptions(def, modelId, auth, providerOpts, stopSequences, maxOutputTokens)
    case 'aws-bedrock':
      return buildBedrockOptions(modelId, auth, stopSequences, maxOutputTokens)
    case 'vertex-ai':
      if (def.id === 'google-vertex-anthropic') {
        return buildVertexAnthropicOptions(modelId, auth, providerOpts, stopSequences, maxOutputTokens)
      }
      return buildVertexGeminiOptions(modelId, auth, providerOpts, stopSequences, maxOutputTokens)
    case 'google-ai':
      return buildGoogleAIOptions(modelId, auth, stopSequences, maxOutputTokens)
    default:
      logger.warn(`[Provider] Unsupported BAML provider type: ${def.bamlProvider}`)
      return undefined
  }
}

/** Anthropic direct API — api-key or OAuth */
function buildAnthropicOptions(modelId: string, auth: AuthInfo | null, stopSequences?: string[], maxOutputTokens?: number): Record<string, any> | undefined {
  const base: Record<string, any> = {
    model: modelId,
    max_tokens: maxOutputTokens ?? 8192,
    temperature: 0.7,
    allowed_role_metadata: ['cache_control'],
    ...(stopSequences && stopSequences.length > 0 ? { stop_sequences: stopSequences } : {}),
  }

  if (auth?.type === 'api') {
    base.api_key = auth.key
  } else if (auth?.type === 'oauth') {
    base.headers = {
      'Authorization': `Bearer ${auth.accessToken}`,
      'anthropic-beta': 'oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14',
      'X-API-Key': '',
    }
  } else {
    // Fall back to env var
    const envKey = process.env.ANTHROPIC_API_KEY
    if (!envKey) return undefined
    base.api_key = envKey
  }

  return base
}

/** OpenAI direct API — api-key or OAuth (Codex endpoint) */
function buildOpenAIOptions(modelId: string, auth: AuthInfo | null, stopSequences?: string[], maxOutputTokens?: number): Record<string, any> | undefined {
  const base: Record<string, any> = {
    model: modelId,
    ...(maxOutputTokens ? { max_tokens: maxOutputTokens } : {}),
    ...(stopSequences && stopSequences.length > 0 ? { stop: stopSequences } : {}),
  }

  if (auth?.type === 'api') {
    base.api_key = auth.key
  } else if (auth?.type === 'oauth') {
    // ChatGPT subscription uses the Codex responses endpoint
    base.api_key = auth.accessToken
    base.base_url = 'https://chatgpt.com/backend-api/codex'
    base.headers = {
      ...(auth.accountId ? { 'ChatGPT-Account-Id': auth.accountId } : {}),
    }
  } else {
    const envKey = process.env.OPENAI_API_KEY
    if (!envKey) return undefined
    base.api_key = envKey
  }

  return base
}

/** OpenAI Responses API — ChatGPT subscription (Codex endpoint) or direct API */
// NOTE: The Responses API (/v1/responses) does NOT support stop sequences or any equivalent parameter.
// Unlike Chat Completions, there is no `stop` field. Passing it causes a 400 error.
// Stop sequences are also excluded in the direct HTTP path in model-proxy.ts#transformForResponsesApi.
function buildOpenAIResponsesOptions(modelId: string, auth: AuthInfo | null, stopSequences?: string[], _maxOutputTokens?: number): Record<string, any> | undefined {
  // NOTE: max_output_tokens not supported — Codex/Copilot Responses API endpoints reject it with 400.
  const base: Record<string, any> = {
    model: modelId,
    // stop sequences intentionally omitted — not supported by Responses API
  }

  if (auth?.type === 'oauth') {
    base.api_key = auth.accessToken
    base.base_url = 'https://chatgpt.com/backend-api/codex'
    base.headers = {
      ...(auth.accountId ? { 'ChatGPT-Account-Id': auth.accountId } : {}),
    }
  } else if (auth?.type === 'api') {
    base.api_key = auth.key
  } else {
    const envKey = process.env.OPENAI_API_KEY
    if (!envKey) return undefined
    base.api_key = envKey
  }

  return base
}

/** OpenAI-generic (OpenRouter, Local, Cerebras, Vercel, GitHub Copilot) */
function buildOpenAIGenericOptions(
  def: ProviderDefinition,
  modelId: string,
  auth: AuthInfo | null,
  providerOpts?: ProviderOptions,
  stopSequences?: string[],
  maxOutputTokens?: number,
): Record<string, any> | undefined {
  // For the local provider, prefer runtime config override
  const localConfig = def.id === 'local' ? getLocalProviderConfig() : null
  const baseUrl = localConfig?.baseUrl ?? providerOpts?.baseUrl ?? def.defaultBaseUrl
  const base: Record<string, any> = {
    model: modelId,
    ...(maxOutputTokens ? { max_tokens: maxOutputTokens } : {}),
    ...(baseUrl ? { base_url: baseUrl } : {}),
    ...(stopSequences && stopSequences.length > 0 ? { stop: stopSequences } : {}),
  }

  // Provider-specific headers
  if (def.id === 'github-copilot') {
    base.headers = {
      'Openai-Intent': 'conversation-edits',
      'x-initiator': 'user',
      'User-Agent': 'GitHubCopilotChat/0.35.0',
      'Editor-Version': 'vscode/1.107.0',
      'Editor-Plugin-Version': 'copilot-chat/0.35.0',
      'Copilot-Integration-Id': 'vscode-chat',
    }
  }

  if (auth?.type === 'api') {
    base.api_key = auth.key
  } else if (auth?.type === 'oauth') {
    // Copilot: accessToken is the exchanged Copilot API token
    base.api_key = auth.accessToken
  } else {
    // Try env vars from auth methods
    for (const method of def.authMethods) {
      if (method.type === 'api-key' && method.envKeys) {
        for (const envKey of method.envKeys) {
          const value = process.env[envKey]
          if (value) {
            base.api_key = value
            return base
          }
        }
      }
    }
    // Local provider doesn't require an API key
    if (def.id === 'local') {
      return base
    }
    return undefined
  }

  return base
}

/** AWS Bedrock — uses AWS credential chain, no explicit api_key */
function buildBedrockOptions(modelId: string, auth: AuthInfo | null, stopSequences?: string[], maxOutputTokens?: number): Record<string, any> | undefined {
  const base: Record<string, any> = {
    model: modelId,
    inference_configuration: {
      max_tokens: maxOutputTokens ?? 8192,
      temperature: 0.7,
    },
    ...(stopSequences && stopSequences.length > 0
      ? { additional_model_request_fields: { stop_sequences: stopSequences } }
      : {}),
    allowed_role_metadata: ['cache_control'],
  }

  if (auth?.type === 'aws') {
    if (auth.region) base.region = auth.region
    if (auth.profile) base.profile = auth.profile
  }

  return base
}

/** Vertex AI — Gemini models on GCP */
function buildVertexGeminiOptions(
  modelId: string,
  auth: AuthInfo | null,
  providerOpts?: ProviderOptions,
  stopSequences?: string[],
  maxOutputTokens?: number,
): Record<string, any> | undefined {
  const location = providerOpts?.location ?? process.env.GOOGLE_CLOUD_LOCATION ?? 'us-east5'
  const base: Record<string, any> = {
    model: modelId,
    location,
    generationConfig: {
      temperature: 0.7,
      ...(maxOutputTokens ? { maxOutputTokens } : {}),
      ...(stopSequences && stopSequences.length > 0 ? { stopSequences } : {}),
    },
  }

  if (auth?.type === 'gcp') {
    base.credentials = auth.credentialsPath
  } else {
    const envCreds = process.env.GOOGLE_APPLICATION_CREDENTIALS
    if (!envCreds) return undefined
    base.credentials = envCreds
  }

  return base
}

/** Vertex AI — Anthropic Claude models on GCP */
function buildVertexAnthropicOptions(
  modelId: string,
  auth: AuthInfo | null,
  providerOpts?: ProviderOptions,
  stopSequences?: string[],
  maxOutputTokens?: number,
): Record<string, any> | undefined {
  const project = providerOpts?.project ?? process.env.GCLOUD_PROJECT ?? process.env.GOOGLE_CLOUD_PROJECT ?? 'magnitudeai'
  const location = providerOpts?.location ?? process.env.GOOGLE_CLOUD_LOCATION ?? 'us-east5'
  const base: Record<string, any> = {
    model: modelId,
    location,
    anthropic_version: 'vertex-2023-10-16',
    max_tokens: maxOutputTokens ?? 8192,
    base_url: `https://${location}-aiplatform.googleapis.com/v1/projects/${project}/locations/${location}/publishers/anthropic/models`,
    ...(stopSequences && stopSequences.length > 0 ? { stop_sequences: stopSequences } : {}),
    allowed_role_metadata: ['cache_control'],
  }

  if (auth?.type === 'gcp') {
    base.credentials = auth.credentialsPath
  } else {
    const envCreds = process.env.GOOGLE_APPLICATION_CREDENTIALS
    if (!envCreds) return undefined
    base.credentials = envCreds
  }

  return base
}


/** Anthropic-compatible providers (e.g. MiniMax) — api-key with custom base_url */
function buildAnthropicCompatibleOptions(
  def: ProviderDefinition,
  modelId: string,
  auth: AuthInfo | null,
  providerOpts?: ProviderOptions,
  stopSequences?: string[],
  maxOutputTokens?: number,
): Record<string, any> | undefined {
  const baseUrl = providerOpts?.baseUrl ?? def.defaultBaseUrl
  const base: Record<string, any> = {
    model: modelId,
    max_tokens: maxOutputTokens ?? 8192,
    temperature: 0.7,
    ...(baseUrl ? { base_url: baseUrl } : {}),
    ...(stopSequences && stopSequences.length > 0 ? { stop_sequences: stopSequences } : {}),
  }

  if (auth?.type === 'api') {
    base.api_key = auth.key
  } else {
    // Try env vars from auth methods
    for (const method of def.authMethods) {
      if (method.type === 'api-key' && method.envKeys) {
        for (const envKey of method.envKeys) {
          const value = process.env[envKey]
          if (value) {
            base.api_key = value
            return base
          }
        }
      }
    }
    return undefined
  }

  return base
}
/** GitHub Copilot with Codex model — uses openai-responses BAML provider */
// NOTE: Also uses the Responses API, which does not support stop sequences (see buildOpenAIResponsesOptions).
function buildCopilotCodexOptions(modelId: string, auth: AuthInfo | null, stopSequences?: string[], _maxOutputTokens?: number): Record<string, any> | undefined {
  // NOTE: max_output_tokens not supported — Codex/Copilot Responses API endpoints reject it with 400.
  if (auth?.type !== 'oauth') return undefined

  return {
    model: modelId,
    // stop sequences intentionally omitted — not supported by Responses API
    api_key: auth.accessToken,
    base_url: 'https://api.githubcopilot.com',
    headers: {
      'Openai-Intent': 'conversation-edits',
      'x-initiator': 'user',
      'User-Agent': 'GitHubCopilotChat/0.35.0',
      'Editor-Version': 'vscode/1.107.0',
      'Editor-Plugin-Version': 'copilot-chat/0.35.0',
      'Copilot-Integration-Id': 'vscode-chat',
    },
  }
}

/** Google AI (Gemini API) — direct API key */
function buildGoogleAIOptions(modelId: string, auth: AuthInfo | null, stopSequences?: string[], maxOutputTokens?: number): Record<string, any> | undefined {
  const base: Record<string, any> = {
    model: modelId,
    generationConfig: {
      temperature: 0.7,
      ...(maxOutputTokens ? { maxOutputTokens } : {}),
      ...(stopSequences && stopSequences.length > 0 ? { stopSequences } : {}),
    },
  }

  if (auth?.type === 'api') {
    base.api_key = auth.key
  } else {
    const envKey = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY
    if (!envKey) return undefined
    base.api_key = envKey
  }

  return base
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/** Deep-merge source into target. Arrays are replaced, not concatenated. */
function deepMerge(target: Record<string, any>, source: Record<string, any>): Record<string, any> {
  const result = { ...target }
  for (const key of Object.keys(source)) {
    if (
      source[key] && typeof source[key] === 'object' && !Array.isArray(source[key]) &&
      result[key] && typeof result[key] === 'object' && !Array.isArray(result[key])
    ) {
      result[key] = deepMerge(result[key], source[key])
    } else {
      result[key] = source[key]
    }
  }
  return result
}
