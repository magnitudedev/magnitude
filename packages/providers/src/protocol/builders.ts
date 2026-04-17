/**
 * Typed option builders — compose capability serializers into final option structs.
 */

import type { AuthInfo, ProviderOptions } from '../types'
import type {
  AnthropicOptions,
  OpenAIOptions,
  OpenAIGenericOptions,
  AnthropicProviderProtocol,
  OpenAIProviderProtocol,
  OpenAIGenericProviderProtocol,
} from './types'
import { resolveAnthropicAuth, resolveOpenAIAuth, resolveOpenAIGenericAuth } from './auth'

export function buildOpenAIGenericOptions(
  protocol: OpenAIGenericProviderProtocol,
  modelId: string,
  auth: AuthInfo | null,
  providerOpts: ProviderOptions | undefined,
  stopSeqs: string[],
  maxTokens: number | undefined,
  grammar: string | undefined,
): OpenAIGenericOptions | undefined {
  const authFields = resolveOpenAIGenericAuth(protocol.authStrategy, auth)
  if (authFields === undefined) return undefined

  const caps = protocol.capabilities
  const baseUrl = providerOpts?.baseUrl ?? protocol.defaultBaseUrl

  const options: OpenAIGenericOptions = {
    model: modelId,
    ...(maxTokens ? { max_tokens: maxTokens } : {}),
    ...(baseUrl ? { base_url: baseUrl } : {}),
    ...authFields,
    ...(stopSeqs.length > 0 ? caps.stopSequences?.(stopSeqs) ?? {} : {}),
    ...(grammar ? caps.grammar?.(grammar) ?? {} : {}),
    ...(caps.reasoningEffort?.(modelId) ?? {}),
    ...(caps.staticOptions ?? {}),
  }

  return options
}

export function buildAnthropicOptions(
  protocol: AnthropicProviderProtocol,
  modelId: string,
  auth: AuthInfo | null,
  providerOpts: ProviderOptions | undefined,
  stopSeqs: string[],
  maxTokens: number | undefined,
): AnthropicOptions | undefined {
  const authFields = resolveAnthropicAuth(protocol.authStrategy, auth)
  if (authFields === undefined) return undefined

  const caps = protocol.capabilities
  const baseUrl = providerOpts?.baseUrl ?? protocol.defaultBaseUrl

  const resolvedMaxTokens = caps.maxTokens ? (caps.maxTokens(maxTokens ?? 8192).max_tokens ?? (maxTokens ?? 8192)) : (maxTokens ?? 8192)
  const options: AnthropicOptions = {
    model: modelId,
    max_tokens: resolvedMaxTokens,
    ...(baseUrl ? { base_url: baseUrl } : {}),
    ...authFields,
    ...(stopSeqs.length > 0 ? caps.stopSequences?.(stopSeqs) ?? {} : {}),
    ...(caps.reasoningEffort?.(modelId) ?? {}),
    ...(caps.staticOptions ?? {}),
  }

  return options
}

export function buildOpenAIOptions(
  protocol: OpenAIProviderProtocol,
  modelId: string,
  auth: AuthInfo | null,
  providerOpts: ProviderOptions | undefined,
  stopSeqs: string[],
  maxTokens: number | undefined,
): OpenAIOptions | undefined {
  const authFields = resolveOpenAIAuth(protocol.authStrategy, auth)
  if (authFields === undefined) return undefined

  const caps = protocol.capabilities

  const options: OpenAIOptions = {
    model: modelId,
    ...(maxTokens && caps.maxTokens ? caps.maxTokens(maxTokens) : {}),
    ...authFields,
    ...(stopSeqs.length > 0 ? caps.stopSequences?.(stopSeqs) ?? {} : {}),
    ...(caps.reasoningEffort?.(modelId) ?? {}),
    ...(caps.staticOptions ?? {}),
  }

  if (typeof providerOpts?.instructions === 'string') options.instructions = providerOpts.instructions
  if (typeof providerOpts?.store === 'boolean') options.store = providerOpts.store

  // Merge custom headers from providerOpts (used by OpenAI Responses API)
  const customHeaders = extractHeaders(providerOpts)
  if (Object.keys(customHeaders).length > 0) {
    options.headers = { ...(options.headers ?? {}), ...customHeaders }
  }

  return options
}

function extractHeaders(providerOpts?: ProviderOptions): Record<string, string> {
  const value = providerOpts?.headers
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const pairs = Object.entries(value as Record<string, unknown>).filter(
    ([, v]) => typeof v === 'string',
  )
  return Object.fromEntries(pairs) as Record<string, string>
}
