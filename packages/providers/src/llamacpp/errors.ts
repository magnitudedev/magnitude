import { Option } from "effect"
import {
  payloadSample,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
  type ProviderCall,
  type ProviderRejection,
  type RejectedHttpResponse,
} from "@magnitudedev/ai"

/**
 * Llama.cpp servers return OpenAI-compatible error responses:
 * { error: { message, type, code, param } }
 *
 * Since Llama.cpp is a local server, the error surface is simpler.
 * There are no credit/billing errors. Common cases:
 * - 401: auth required (if server configured with API key)
 * - 404: model not found
 * - 413 / 400: context limit exceeded
 * - 429: rate limited (some servers)
 * - 500: internal server error
 */

interface LlamaCppErrorBody {
  readonly error: {
    readonly message: string
    readonly type?: string
    readonly code?: string
    readonly param?: string | null
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function tryParseErrorBody(body: string): LlamaCppErrorBody | null {
  try {
    const parsed: unknown = JSON.parse(body)
    if (!isRecord(parsed)) return null
    const error = parsed.error
    if (!isRecord(error) || typeof error.message !== "string") return null
    return {
      error: {
        message: error.message,
        ...(typeof error.type === "string" ? { type: error.type } : {}),
        ...(typeof error.code === "string" ? { code: error.code } : {}),
        ...(error.param !== undefined ? { param: typeof error.param === "string" ? error.param : null } : {}),
      },
    }
  } catch {
    return null
  }
}

function isContextLimit(message: string): boolean {
  const text = message.toLowerCase()
  return [
    "prompt is too long",
    "token count exceeds the maximum",
    "maximum context length",
    "context_length_exceeded",
    "exceeded model token limit",
    "too many tokens",
    "context size",
  ].some((pattern) => text.includes(pattern))
}

function classifyLlamaCppError(
  response: RejectedHttpResponse,
  parsed: LlamaCppErrorBody | null,
): ProviderRejection {
  const message = parsed?.error.message ?? `HTTP ${response.status}`
  const errorType = parsed?.error.type
  const errorCode = parsed?.error.code

  // Auth
  if (response.status === 401 || response.status === 403 || errorType === "authentication_error") {
    return { _tag: "AuthRejected", message }
  }

  // Model not found
  if (response.status === 404 || errorCode === "model_not_found" || errorCode === "model_not_found_error") {
    return { _tag: "ModelUnavailable", message }
  }

  // Rate limited
  if (response.status === 429 || errorType === "rate_limit_error") {
    return {
      _tag: "RateLimited",
      message,
      retryPolicy: {
        retry: true,
        retryAfterMs: response.retryAfterMs !== null ? Option.some(response.retryAfterMs) : Option.none(),
      },
    }
  }

  // Context limit
  if (isContextLimit(message) || response.status === 413) {
    return { _tag: "ContextLimitExceeded", message }
  }

  // Model capability missing (e.g. trying to use vision on a text-only model)
  if (errorCode === "model_not_multimodal" || message.toLowerCase().includes("does not support vision") || message.toLowerCase().includes("does not support images")) {
    return { _tag: "ModelCapabilityMissing", message }
  }

  // Grammar not supported
  if (errorCode === "model_not_grammar_compatible" || message.toLowerCase().includes("grammar")) {
    return { _tag: "ProviderCapabilityMissing", message }
  }

  // Server errors
  if (response.status >= 500) {
    return {
      _tag: "UpstreamFailure",
      message,
      retryPolicy: { retry: true, retryAfterMs: Option.none() },
    }
  }

  // Default: invalid request
  return { _tag: "InvalidRequest", message }
}

export function classifyLlamaCppRejectedResponse(
  call: ProviderCall,
  response: RejectedHttpResponse,
): StreamStartProviderRejection | StreamStartProviderCorrectnessViolation {
  const parsed = tryParseErrorBody(response.body)

  // If the body is non-empty but can't be parsed, it's a correctness violation.
  // Empty body on error status is acceptable (some Llama.cpp servers do this).
  if (parsed === null && response.body.trim().length > 0) {
    return new StreamStartProviderCorrectnessViolation({
      call,
      response,
      violation: {
        _tag: "InvalidErrorEnvelope",
        status: response.status,
        body: payloadSample(response.body),
        issue: { message: "Llama.cpp error response did not match the expected OpenAI error envelope shape" },
      },
    })
  }

  return new StreamStartProviderRejection({
    call,
    response,
    rejection: classifyLlamaCppError(response, parsed),
  })
}
