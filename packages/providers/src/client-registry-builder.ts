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
import { buildAnthropicOptions, buildOpenAIOptions, buildOpenAIGenericOptions } from './protocol/builders'
import type { AnthropicOptions, OpenAIOptions, OpenAIGenericOptions, ProviderProtocol } from './protocol/types'
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

  const { bamlProvider, protocol } = def.resolveProtocol(auth)
  const modelDef = def.models.find(m => m.id === modelId)
  const maxOutputTokens = maxTokensOverride ?? modelDef?.maxOutputTokens

  const options = buildOptions(protocol, modelId, auth, providerOptions, stopSequences, maxOutputTokens, grammar)
  if (!options) return undefined

  const cr = new ClientRegistry()
  cr.addLlmClient(CHAT_CLIENT_NAME, bamlProvider, options as Record<string, any>)
  cr.setPrimary(CHAT_CLIENT_NAME)
  return cr
}

// ---------------------------------------------------------------------------
// Option builder — delegates to typed protocol builders
// ---------------------------------------------------------------------------

function buildOptions(
  protocol: ProviderProtocol,
  modelId: string,
  auth: AuthInfo | null,
  providerOpts?: ProviderOptions,
  stopSequences?: string[],
  maxOutputTokens?: number,
  grammar?: string,
): AnthropicOptions | OpenAIOptions | OpenAIGenericOptions | undefined {
  const stopSeqs = stopSequences ?? []

  switch (protocol.type) {
    case 'anthropic':
      return buildAnthropicOptions(protocol, modelId, auth, providerOpts, stopSeqs, maxOutputTokens)
    case 'openai':
      return buildOpenAIOptions(protocol, modelId, auth, providerOpts, stopSeqs, maxOutputTokens)
    case 'openai-generic':
      return buildOpenAIGenericOptions(protocol, modelId, auth, providerOpts, stopSeqs, maxOutputTokens, grammar)
  }
}

export function __testOnly_buildProviderOptions(
  providerId: string,
  modelId: string,
  auth: AuthInfo | null,
  providerOptions?: ProviderOptions,
  stopSequences?: string[],
  grammar?: string,
): AnthropicOptions | OpenAIOptions | OpenAIGenericOptions | undefined {
  const def = getProvider(providerId)
  if (!def) return undefined

  const { protocol } = def.resolveProtocol(auth)
  const modelDef = def.models.find(m => m.id === modelId)
  return buildOptions(
    protocol,
    modelId,
    auth,
    providerOptions,
    stopSequences,
    modelDef?.maxOutputTokens,
    grammar,
  )
}
