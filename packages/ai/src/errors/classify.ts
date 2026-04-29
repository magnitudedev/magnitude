import type { HttpConnectionFailure, StreamFailure } from "./failure"
import type { ConnectionError, StreamError } from "./model-error"
import {
  AuthFailed,
  ContextLimitExceeded,
  InvalidRequest,
  ParseError,
  RateLimited,
  TransportError,
  UsageLimitExceeded,
} from "./model-error"

const CONTEXT_LIMIT_PATTERNS = [
  "prompt is too long",
  "token count exceeds the maximum",
  "maximum context length",
  "context_length_exceeded",
  "exceeded model token limit",
]

const AUTH_SIGNAL_PATTERNS = [
  "missing_scope",
  "insufficient_scope",
  "invalid_token",
  "token expired",
  "expired token",
  "unauthorized",
  "forbidden",
  "authentication",
]

const USAGE_LIMIT_PATTERNS = [
  "usage_limit_exceeded",
  "billing_hard_limit_reached",
  "insufficient_quota",
  "quota exceeded",
  "credit balance is too low",
]

function hasPattern(text: string, patterns: readonly string[]): boolean {
  return patterns.some((pattern) => text.includes(pattern))
}

function parseRetryAfterMs(headers: Headers): number | null {
  const value = headers.get("retry-after")
  if (value == null) return null

  const seconds = Number(value)
  if (Number.isFinite(seconds)) return seconds * 1000

  const date = Date.parse(value)
  if (Number.isFinite(date)) return Math.max(0, date - Date.now())

  return null
}

function tryParseJsonObject(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text)
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    // fall through
  }

  let offset = 0
  while ((offset = text.indexOf("{", offset)) !== -1) {
    try {
      const parsed = JSON.parse(text.slice(offset))
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      // try next object start
    }
    offset += 1
  }

  return null
}

function getNestedErrorObject(text: string): Record<string, unknown> | null {
  const parsed = tryParseJsonObject(text)
  const error = parsed?.error
  return error && typeof error === "object" && !Array.isArray(error)
    ? (error as Record<string, unknown>)
    : null
}

function getErrorMessage(text: string): string {
  const message = getNestedErrorObject(text)?.message
  return typeof message === "string" ? message : text
}

function getErrorCode(text: string): string | null {
  const code = getNestedErrorObject(text)?.code
  return typeof code === "string" ? code : null
}

export function defaultClassifyConnectionError(
  sourceId: string,
  failure: HttpConnectionFailure,
): ConnectionError {
  const body = failure.body
  const lower = body.toLowerCase()

  if (hasPattern(lower, CONTEXT_LIMIT_PATTERNS)) {
    return new ContextLimitExceeded({ sourceId, status: failure.status, message: getErrorMessage(body) })
  }

  if (failure.status === 401 || failure.status === 403 || hasPattern(lower, AUTH_SIGNAL_PATTERNS)) {
    return new AuthFailed({ sourceId, status: failure.status, message: getErrorMessage(body) })
  }

  if (failure.status === 429) {
    const code = getErrorCode(body)?.toLowerCase()
    if ((code && hasPattern(code, USAGE_LIMIT_PATTERNS)) || hasPattern(lower, USAGE_LIMIT_PATTERNS)) {
      return new UsageLimitExceeded({ sourceId, status: failure.status, message: getErrorMessage(body) })
    }

    return new RateLimited({
      sourceId,
      status: failure.status,
      message: getErrorMessage(body),
      retryAfterMs: parseRetryAfterMs(failure.headers),
    })
  }

  if (failure.status >= 400 && failure.status < 500) {
    return new InvalidRequest({ sourceId, status: failure.status, message: getErrorMessage(body) })
  }

  return new TransportError({
    sourceId,
    status: failure.status,
    message: getErrorMessage(body),
    retryable: failure.status >= 500,
  })
}

export function defaultClassifyStreamError(
  sourceId: string,
  failure: StreamFailure,
): StreamError {
  switch (failure._tag) {
    case "ReadFailure":
      return new TransportError({ sourceId, status: null, message: String(failure.cause), retryable: true })
    case "SseParseFailure":
      return new ParseError({ sourceId, message: `SSE parse failure: ${failure.payload}` })
    case "ChunkDecodeFailure":
      return new ParseError({ sourceId, message: `Chunk decode failure: ${String(failure.cause)}` })
  }
}
