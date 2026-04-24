import {
  AuthFailed,
  ContextLimitExceeded,
  ParseError as ProviderParseError,
  SubscriptionRequired,
  TransportError as ProviderTransportError,
  UsageLimitExceeded,
} from '@magnitudedev/providers'
import type { ModelError } from '@magnitudedev/providers'
import { BamlClientHttpError, BamlValidationError } from '@magnitudedev/llm-core'

export type NonRetryableReason = 'context-limit' | 'auth' | 'parse' | 'client-error' | 'not-configured' | 'disconnected' | null

export function authReconnectMessage(providerName?: string | null): string {
  return providerName
    ? `${providerName} session expired or became invalid. Please reconnect in /settings.`
    : 'Your provider session expired or became invalid. Please reconnect in /settings.'
}

export function isAuthReconnectMessage(message: string): boolean {
  const lower = message.toLowerCase()
  return lower.includes('session expired or became invalid') && lower.includes('/settings')
}

export function resolveFailureMessage(error: ModelError): string {
  if (error._tag === 'NotConfigured') {
    return 'No model configured. Please connect a provider and select a model in /settings.'
  }
  if (error._tag === 'ProviderDisconnected') {
    return error.message
  }
  if (error._tag === 'AuthFailed') {
    return authReconnectMessage()
  }
  if (error._tag === 'SubscriptionRequired') {
    return error.message
  }
  if (error._tag === 'UsageLimitExceeded') {
    return error.message
  }
  return `Authentication failed: ${error.message}`
}

/**
 * Build the error message and errorCode for the general TurnError fallback path (Path 3).
 * Extracts errorCode from the cause for CTA rendering and truncates long messages.
 * Uses resolveFailureMessage for known ModelError types to avoid noisy prefixes.
 */
export function buildGeneralErrorPayload(errorMessage: string, errorCause: unknown): {
  message: string
  errorCode: string | undefined
} {
  // Extract errorCode from the cause for CTA rendering (e.g., usage_limit_exceeded_weekly)
  const errorCode = errorCause !== null && typeof errorCause === 'object' && 'code' in errorCause
    ? (errorCause as any).code as string | undefined
    : undefined

  // If the cause has a known _tag, use its clean message instead of the raw error text
  const hasTag = typeof errorCause === 'object' && errorCause !== null && '_tag' in errorCause
  const resolved = hasTag ? resolveFailureMessage(errorCause as ModelError) : null

  let message: string
  if (resolved) {
    message = resolved
  } else {
    // Truncate and prefix only for unclassified errors
    const suffix = 'Unexpected error while executing turn: '
    const maxLen = 500
    const truncated = (suffix + errorMessage).length > maxLen
      ? (suffix + errorMessage).slice(0, maxLen) + '...'
      : suffix + errorMessage
    message = truncated
  }

  return { message, errorCode }
}

export function classifyRetryability(error: unknown): NonRetryableReason {
  if (error instanceof ContextLimitExceeded) return 'context-limit'
  if (error instanceof AuthFailed) return 'auth'
  if (error instanceof SubscriptionRequired) return 'client-error'
  if (error instanceof UsageLimitExceeded) return 'client-error'
  if (error instanceof ProviderParseError) return 'parse'
  if (error instanceof ProviderTransportError) {
    const s = error.status
    if (s !== null && s >= 400 && s < 500 && s !== 408 && s !== 429) return 'client-error'
    return null
  }
  if (error instanceof BamlClientHttpError) {
    const s = error.status_code
    if (s !== undefined && s >= 400 && s < 500 && s !== 408 && s !== 429) return 'client-error'
    return null
  }
  if (error instanceof BamlValidationError) return 'parse'
  return null
}
