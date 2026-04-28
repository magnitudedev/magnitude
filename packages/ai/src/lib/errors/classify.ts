import {
  AuthFailed,
  ContextLimitExceeded,
  InvalidRequest,
  RateLimited,
  type ModelError,
  TransportError,
  UsageLimitExceeded,
} from "./model-error"

export type ErrorClassifier = (error: unknown, provider: { readonly id: string; readonly name: string }) => ModelError

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

function parseRetryAfterMs(message: string): number | null {
  const match = message.match(/retry[- ]after[:=\s]+(\d+)\s*(ms|s|sec|secs|second|seconds)?/i)
  if (!match) return null

  const value = Number(match[1])
  if (!Number.isFinite(value)) return null

  const unit = match[2]?.toLowerCase()
  return unit === "ms" ? value : value * 1000
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

function getErrorMessage(text: string): string | null {
  const message = getNestedErrorObject(text)?.message
  return typeof message === "string" ? message : null
}

function getErrorCode(text: string): string | null {
  const code = getNestedErrorObject(text)?.code
  return typeof code === "string" ? code : null
}

function classifyHttpError(
  status: number,
  message: string,
  providerId: string,
): ModelError {
  const lower = message.toLowerCase()

  if (hasPattern(lower, CONTEXT_LIMIT_PATTERNS)) {
    return new ContextLimitExceeded({
      providerId,
      status,
      message,
    })
  }

  if (status === 401 || status === 403 || hasPattern(lower, AUTH_SIGNAL_PATTERNS)) {
    return new AuthFailed({
      providerId,
      status,
      message,
    })
  }

  if (status === 429) {
    const code = getErrorCode(message)?.toLowerCase()
    if ((code && hasPattern(code, USAGE_LIMIT_PATTERNS)) || hasPattern(lower, USAGE_LIMIT_PATTERNS)) {
      return new UsageLimitExceeded({
        providerId,
        status,
        message: getErrorMessage(message) ?? message,
      })
    }

    return new RateLimited({
      providerId,
      status,
      message,
      retryAfterMs: parseRetryAfterMs(message),
    })
  }

  if (status >= 400 && status < 500) {
    return new InvalidRequest({
      providerId,
      status,
      message,
    })
  }

  return new TransportError({
    providerId,
    status,
    message,
    retryable: status >= 500,
  })
}

function isStatusLike(value: unknown): value is { readonly status: number; readonly message?: string } {
  return typeof value === "object" && value !== null && "status" in value && typeof value.status === "number"
}

function getErrorMessageFromUnknown(error: unknown): string {
  if (typeof error === "string") return error
  if (error instanceof Error) return error.message
  if (typeof error === "object" && error !== null && "message" in error && typeof error.message === "string") {
    return error.message
  }
  return String(error)
}

export const classifyGenericError: ErrorClassifier = (error, provider) => {
  if (isStatusLike(error)) {
    return classifyHttpError(error.status, error.message ?? "", provider.id)
  }

  return new TransportError({
    providerId: provider.id,
    status: null,
    message: getErrorMessageFromUnknown(error),
    retryable: true,
  })
}
