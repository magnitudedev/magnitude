import {
  AuthFailed,
  ContextLimitExceeded,
  RateLimited,
  TransportError,
  type ModelError,
} from './model-error'

const CONTEXT_LIMIT_PATTERNS = [
  'prompt is too long',
  'token count exceeds the maximum',
  'maximum context length',
  'context_length_exceeded',
]

function hasContextLimitSignal(text: string): boolean {
  return CONTEXT_LIMIT_PATTERNS.some(pattern => text.includes(pattern))
}

function parseRetryAfterMs(message: string): number | null {
  const match = message.match(/retry[- ]after[:=\s]+(\d+)\s*(ms|s|sec|secs|second|seconds)?/i)
  if (!match) return null
  const value = Number(match[1])
  if (!Number.isFinite(value)) return null
  const unit = match[2]?.toLowerCase()
  return unit === 'ms' ? value : value * 1000
}

/**
 * Classify an HTTP error by status code and message text.
 * This is the shared heuristic layer — driver-agnostic.
 */
export function classifyHttpError(status: number, message: string): ModelError {
  const lower = message.toLowerCase()
  if (hasContextLimitSignal(lower)) {
    return new ContextLimitExceeded({ message })
  }
  if (status === 401 || status === 403) {
    return new AuthFailed({ message })
  }
  if (status === 429) {
    return new RateLimited({ message, retryAfterMs: parseRetryAfterMs(message) })
  }
  return new TransportError({ message, status })
}

/**
 * Fallback for non-HTTP errors.
 */
export function classifyUnknownError(error: unknown): ModelError {
  return new TransportError({
    message: error instanceof Error ? error.message : String(error),
    status: null,
  })
}

/**
 * Determine whether a ModelError is retryable for connection-level retry.
 *
 * Retryable (transient):
 *   - TransportError with 5xx, 408 (timeout), 429 (rate limit), or unknown status
 *   - RateLimited
 *
 * Non-retryable (permanent or semantic):
 *   - TransportError with 4xx client errors (except 408, 429)
 *   - ContextLimitExceeded
 *   - AuthFailed
 *   - NotConfigured
 *   - ProviderDisconnected
 *   - ParseError
 */
export function isRetryableError(error: ModelError): boolean {
  switch (error._tag) {
    case 'RateLimited':
      return true
    case 'TransportError': {
      const { status } = error
      if (status === null) return true // unknown status — assume transient
      if (status === 408 || status === 429) return true
      if (status >= 400 && status < 500) return false // other 4xx — client error
      return true // 5xx, etc.
    }
    default:
      return false
  }
}