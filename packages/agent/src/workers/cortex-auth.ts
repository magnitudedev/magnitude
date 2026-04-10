import {
  AuthFailed,
  ContextLimitExceeded,
  ParseError as ProviderParseError,
  TransportError as ProviderTransportError,
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
  return `Authentication failed: ${error.message}`
}

export function classifyRetryability(error: unknown): NonRetryableReason {
  if (error instanceof ContextLimitExceeded) return 'context-limit'
  if (error instanceof AuthFailed) return 'auth'
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
