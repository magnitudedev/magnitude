import {
  AuthFailed,
  ContextLimitExceeded,
  RateLimited,
  SubscriptionRequired,
  TransportError,
  UsageLimitExceeded,
  type ModelError,
} from './model-error'

const CONTEXT_LIMIT_PATTERNS = [
  'prompt is too long',
  'token count exceeds the maximum',
  'maximum context length',
  'context_length_exceeded',
  'exceeded model token limit',
]

const AUTH_SIGNAL_PATTERNS = [
  'missing_scope',
  'insufficient_scope',
  'invalid_token',
  'token expired',
  'expired token',
  'unauthorized',
  'forbidden',
  'authentication',
]

function hasContextLimitSignal(text: string): boolean {
  return CONTEXT_LIMIT_PATTERNS.some(pattern => text.includes(pattern))
}

function hasAuthSignal(text: string): boolean {
  return AUTH_SIGNAL_PATTERNS.some(pattern => text.includes(pattern))
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
 * Try to find and parse a JSON object within mixed text.
 * First attempts direct JSON.parse on the full text, then
 * scans for `{` positions and tries slices if direct parse fails.
 */
function tryParseJson(text: string): Record<string, unknown> | null {
  // Try direct parse first (pure JSON text)
  try {
    const parsed = JSON.parse(text)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    // Fall through to scanning
  }

  // Scan for JSON objects embedded in text
  let idx = 0
  while ((idx = text.indexOf('{', idx)) !== -1) {
    try {
      const slice = text.slice(idx)
      const parsed = JSON.parse(slice)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      // Try next occurrence
    }
    idx++
  }

  return null
}

function tryParseErrorCode(text: string): string | null {
  const parsed = tryParseJson(text)
  const code = parsed?.error
  if (code && typeof code === 'object') {
    const c = (code as Record<string, unknown>).code
    return typeof c === 'string' ? c : null
  }
  return null
}

function tryParseErrorMessage(text: string): string | null {
  const parsed = tryParseJson(text)
  const msg = parsed?.error
  if (msg && typeof msg === 'object') {
    const m = (msg as Record<string, unknown>).message
    return typeof m === 'string' ? m : null
  }
  return null
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
  if (status === 401 || status === 403 || hasAuthSignal(lower)) {
    return new AuthFailed({ message })
  }
  if (status === 402) {
    const code = tryParseErrorCode(message) ?? 'subscription_required'
    const userMessage = tryParseErrorMessage(message) ?? message
    return new SubscriptionRequired({ message: userMessage, code })
  }
  if (status === 429) {
    const code = tryParseErrorCode(message)
    if (code?.startsWith('usage_limit_exceeded')) {
      const userMessage = tryParseErrorMessage(message) ?? message
      return new UsageLimitExceeded({ message: userMessage, code })
    }
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
 *   - SubscriptionRequired
 *   - UsageLimitExceeded
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
