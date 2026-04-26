import {
  AuthFailed,
  ContextLimitExceeded,
  ParseError as ProviderParseError,
  RateLimited,
  SubscriptionRequired,
  TransportError as ProviderTransportError,
  UsageLimitExceeded,
} from '@magnitudedev/providers'
import type { ModelError } from '@magnitudedev/providers'
import { BamlClientHttpError, BamlValidationError } from '@magnitudedev/llm-core'
import type { TurnOutcome } from '../events'

export type NonRetryableReason = 'context-limit' | 'auth' | 'parse' | 'client-error' | null

export function authReconnectMessage(providerName?: string | null): string {
  return providerName
    ? `${providerName} session expired or became invalid. Please reconnect in /settings.`
    : 'Your provider session expired or became invalid. Please reconnect in /settings.'
}

function truncateUnexpectedError(message: string): string {
  const maxLen = 500
  return message.length > maxLen ? `${message.slice(0, maxLen)}...` : message
}

export function classifyModelError(error: ModelError): TurnOutcome {
  switch (error._tag) {
    case 'NotConfigured':
      return { _tag: 'ProviderNotReady', detail: { _tag: 'NotConfigured' } }
    case 'ProviderDisconnected':
      return {
        _tag: 'ProviderNotReady',
        detail: {
          _tag: 'ProviderDisconnected',
          providerId: error.providerId,
          providerName: error.providerName,
        },
      }
    case 'AuthFailed':
      return {
        _tag: 'ProviderNotReady',
        detail: {
          _tag: 'AuthFailed',
          providerId: error.providerId,
          providerName: error.providerName,
        },
      }
    case 'SubscriptionRequired':
      return {
        _tag: 'ProviderNotReady',
        detail: {
          _tag: 'MagnitudeBilling',
          reason: { _tag: 'SubscriptionRequired', message: error.message },
        },
      }
    case 'UsageLimitExceeded':
      return {
        _tag: 'ProviderNotReady',
        detail: {
          _tag: 'MagnitudeBilling',
          reason: { _tag: 'UsageLimitExceeded', message: error.message },
        },
      }
    case 'ContextLimitExceeded':
      return { _tag: 'ContextWindowExceeded' }
    case 'RateLimited':
      return {
        _tag: 'ConnectionFailure',
        detail: { _tag: 'ProviderError', httpStatus: 429 },
      }
    case 'TransportError':
      return error.status !== null && (error.status === 408 || error.status === 429 || error.status >= 500)
        ? { _tag: 'ConnectionFailure', detail: { _tag: 'ProviderError', httpStatus: error.status } }
        : { _tag: 'ConnectionFailure', detail: { _tag: 'TransportError', ...(error.status !== null ? { httpStatus: error.status } : {}) } }
    case 'ParseError':
      return {
        _tag: 'UnexpectedError',
        message: truncateUnexpectedError(`Provider returned unparseable response: ${error.message}`),
        detail: { _tag: 'CortexDefect' },
      }
  }
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
  if (error instanceof RateLimited) return null
  return null
}
