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
import { getLowestEffortOptions } from './reasoning-effort'
import type { AuthInfo, BamlProviderType, ProviderDefinition, ProviderOptions } from './types'

/** The BAML client name used by CodingAgentChat */
const CHAT_CLIENT_NAME = 'ChatClientNoRetryAnthropicOnly'

function resolveBamlProvider(providerId: string, modelId: string, auth: AuthInfo | null): BamlProviderType {
  const def = getProvider(providerId)
  if (!def) {
    logger.warn(`[Provider] Unknown provider: ${providerId}`)
    return 'anthropic'
  }

  return (providerId === 'openai' && (auth?.type === 'oauth' || auth?.type === 'api'))
    ? 'openai-responses'
    : def.bamlProvider
}

/**
 * Build a ClientRegistry targeting the given provider + model.
 *
 * Returns undefined if no valid auth is available (falls back to BAML static client).
 */
export function buildClientRegistry(
  providerId: string,
  modelId: string,
  auth: AuthInfo | null,
  providerOptions?: ProviderOptions,
  stopSequences?: string[],
  grammar?: string,
  maxTokensOverride?: number,
): ClientRegistry | undefined {
  const def = getProvider(providerId)
  if (!def) {
    logger.warn(`[Provider] Unknown provider: ${providerId}`)
    return undefined
  }

  const providerOpts = providerOptions
  const bamlProvider = resolveBamlProvider(providerId, modelId, auth)

  const modelDef = def.models.find(m => m.id === modelId)
  const maxOutputTokens = maxTokensOverride ?? modelDef?.maxOutputTokens

  let options = buildOptions({ ...def, bamlProvider }, modelId, auth, providerOpts, stopSequences, maxOutputTokens, grammar)
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
  grammar?: string,
): Record<string, any> | undefined {
  switch (def.bamlProvider) {
    case 'anthropic':
      if (def.defaultBaseUrl) {
        return buildAnthropicCompatibleOptions(def, modelId, auth, providerOpts, stopSequences, maxOutputTokens)
      }
      return buildAnthropicOptions(modelId, auth, stopSequences, maxOutputTokens)
    case 'openai':
      return buildOpenAIOptions(modelId, auth, stopSequences, maxOutputTokens)
    case 'openai-responses':
      return buildOpenAIResponsesOptions(modelId, auth, providerOpts, stopSequences, maxOutputTokens)
    case 'openai-generic':
      return buildOpenAIGenericOptions(def, modelId, auth, providerOpts, stopSequences, maxOutputTokens, grammar)
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
function buildOpenAIResponsesOptions(
  modelId: string,
  auth: AuthInfo | null,
  providerOpts?: ProviderOptions,
  stopSequences?: string[],
  _maxOutputTokens?: number,
): Record<string, any> | undefined {
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
      ...extractOpenAIResponsesHeaders(providerOpts),
    }
    base.store = false
  } else if (auth?.type === 'api') {
    base.api_key = auth.key
    const headers = extractOpenAIResponsesHeaders(providerOpts)
    if (Object.keys(headers).length > 0) base.headers = headers
  } else {
    const envKey = process.env.OPENAI_API_KEY
    if (!envKey) return undefined
    base.api_key = envKey
    const headers = extractOpenAIResponsesHeaders(providerOpts)
    if (Object.keys(headers).length > 0) base.headers = headers
  }

  return mergeOpenAIResponsesOverrides(base, providerOpts)
}

/** OpenAI-generic (OpenRouter, Local, Cerebras, Vercel, etc.) */
function buildOpenAIGenericOptions(
  def: ProviderDefinition,
  modelId: string,
  auth: AuthInfo | null,
  providerOpts?: ProviderOptions,
  stopSequences?: string[],
  maxOutputTokens?: number,
  grammar?: string,
): Record<string, any> | undefined {
  const baseUrl = providerOpts?.baseUrl ?? def.defaultBaseUrl
  const base: Record<string, any> = {
    model: modelId,
    ...(maxOutputTokens ? { max_tokens: maxOutputTokens } : {}),
    ...(baseUrl ? { base_url: baseUrl } : {}),
    ...(stopSequences && stopSequences.length > 0 ? { stop: stopSequences } : {}),
    ...(grammar ? { response_format: { type: 'grammar', grammar } } : {}),
    stream_options: { include_usage: true },
  }

  if (def.id === 'magnitude') {
    base.stream = true
  }

  if (auth?.type === 'api') {
    base.api_key = auth.key
  } else if (auth?.type === 'oauth') {
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
    // Local-family / authless providers can run without API keys.
    if (def.providerFamily === 'local' || def.authMethods.some((method) => method.type === 'none')) {
      return base
    }
    return undefined
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
function extractOpenAIResponsesHeaders(providerOpts?: ProviderOptions): Record<string, string> {
  const value = providerOpts?.headers
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const pairs = Object.entries(value as Record<string, unknown>).filter(
    ([, v]) => typeof v === 'string',
  )
  return Object.fromEntries(pairs) as Record<string, string>
}

function mergeOpenAIResponsesOverrides(
  base: Record<string, any>,
  providerOpts?: ProviderOptions,
): Record<string, any> {
  if (!providerOpts) return base
  const overrides: Record<string, unknown> = {}

  if (typeof providerOpts.instructions === 'string') {
    overrides.instructions = providerOpts.instructions
  }
  if (typeof providerOpts.store === 'boolean') {
    overrides.store = providerOpts.store
  }

  if (Object.keys(overrides).length === 0) return base
  return deepMerge(base, overrides)
}

export function __testOnly_buildProviderOptions(
  providerId: string,
  modelId: string,
  auth: AuthInfo | null,
  providerOptions?: ProviderOptions,
  stopSequences?: string[],
  grammar?: string,
): Record<string, any> | undefined {
  const def = getProvider(providerId)
  if (!def) return undefined

  const bamlProvider = resolveBamlProvider(providerId, modelId, auth)

  const modelDef = def.models.find(m => m.id === modelId)
  return buildOptions(
    { ...def, bamlProvider },
    modelId,
    auth,
    providerOptions,
    stopSequences,
    modelDef?.maxOutputTokens,
    grammar,
  )
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
