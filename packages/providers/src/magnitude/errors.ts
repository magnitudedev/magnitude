import { Option } from "effect"
import {
  payloadSample,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
  type ProviderCall,
  type ProviderRejection,
  type RejectedHttpResponse,
} from "@magnitudedev/ai"
import type {
  BillingWindowBudget,
  MagnitudeApiError,
  MagnitudeErrorCode,
  MagnitudeErrorType,
  SubscriptionRequiredDetails,
  UsageLimitDetails,
} from "./contract"

const ERROR_TYPES: readonly MagnitudeErrorType[] = [
  "invalid_request_error",
  "authentication_error",
  "insufficient_quota",
  "rate_limit_error",
  "server_error",
  "service_unavailable",
]

const ERROR_CODES: readonly MagnitudeErrorCode[] = [
  "invalid_api_key",
  "invalid_body",
  "unsupported_field",
  "unsupported_n",
  "invalid_image_url",
  "invalid_multimodal_role",
  "model_not_found",
  "model_not_multimodal",
  "model_not_grammar_compatible",
  "subscription_required",
  "usage_limit_exceeded_five_hour",
  "usage_limit_exceeded_weekly",
  "usage_limit_exceeded_monthly",
  "provider_rate_limited",
  "internal_server_error",
  "provider_error",
  "invariant_violation",
  "upstream_unavailable",
  "stream_interrupted",
]

type MagnitudeErrorBase = Omit<MagnitudeApiError["error"], "code" | "details">
type UsageLimitCode = Extract<MagnitudeErrorCode, `usage_limit_exceeded_${string}`>
type SubscriptionRequiredCode = Extract<MagnitudeErrorCode, "subscription_required">

const USAGE_LIMIT_CODES: readonly UsageLimitCode[] = [
  "usage_limit_exceeded_five_hour",
  "usage_limit_exceeded_weekly",
  "usage_limit_exceeded_monthly",
]

function isUsageLimitCode(code: MagnitudeErrorCode): code is UsageLimitCode {
  return USAGE_LIMIT_CODES.includes(code as UsageLimitCode)
}

export type ParsedMagnitudeApiError =
  | {
      readonly error: MagnitudeErrorBase & {
        readonly code: SubscriptionRequiredCode
        readonly details: SubscriptionRequiredDetails
      }
    }
  | {
      readonly error: MagnitudeErrorBase & {
        readonly code: UsageLimitCode
        readonly details: UsageLimitDetails
      }
    }
  | {
      readonly error: MagnitudeErrorBase & {
        readonly code: Exclude<MagnitudeErrorCode, UsageLimitCode | SubscriptionRequiredCode>
      }
    }

function retryPolicy(retry: boolean, retryAfterMs: number | null): {
  readonly retry: boolean
  readonly retryAfterMs: Option.Option<number>
} {
  return {
    retry,
    retryAfterMs: retryAfterMs === null ? Option.none() : Option.some(retryAfterMs),
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function isErrorType(value: unknown): value is MagnitudeErrorType {
  return typeof value === "string" && ERROR_TYPES.includes(value as MagnitudeErrorType)
}

function isErrorCode(value: unknown): value is MagnitudeErrorCode {
  return typeof value === "string" && ERROR_CODES.includes(value as MagnitudeErrorCode)
}

function isBillingWindowBudget(value: unknown): value is BillingWindowBudget {
  return isRecord(value)
    && typeof value.limitCents === "number"
    && typeof value.usedCents === "number"
    && typeof value.remainingCents === "number"
    && typeof value.windowStart === "string"
    && typeof value.windowEnd === "string"
    && typeof value.remainingMs === "number"
}

function isUsageLimitDetails(value: unknown): value is UsageLimitDetails {
  const windows = isRecord(value) && isRecord(value.windows) ? value.windows : null
  const violatedWindow = isRecord(value) ? value.violatedWindow : null
  if (!isRecord(value)
    || value.category !== "usage_limit_exceeded"
    || typeof violatedWindow !== "string"
    || !["five_hour", "weekly", "monthly"].includes(violatedWindow)
    || windows === null
    || !isBillingWindowBudget(windows[violatedWindow])
    || !isBillingWindowBudget(value.violatedBudget)) return false

  return Object.values(windows).every(isBillingWindowBudget)
}

function isSubscriptionRequiredDetails(value: unknown): value is SubscriptionRequiredDetails {
  return isRecord(value)
    && value.category === "subscription_required"
    && value.requiredPlanId === "pro"
}

export function tryParseErrorBody(body: string): ParsedMagnitudeApiError | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(body)
  } catch {
    return null
  }

  if (!isRecord(parsed)) return null
  const error = parsed.error
  if (!isRecord(error)) return null
  if (typeof error.message !== "string" || error.message.trim().length === 0) return null
  if (!isErrorType(error.type)) return null
  if (!isErrorCode(error.code)) return null
  if (error.param !== null && typeof error.param !== "string") return null

  const base: MagnitudeErrorBase = {
    message: error.message,
    type: error.type,
    param: error.param,
  }

  if (isUsageLimitCode(error.code)) {
    if (!isUsageLimitDetails(error.details)) return null
    return {
      error: {
        ...base,
        code: error.code,
        details: error.details,
      },
    }
  }

  if (error.code === "subscription_required") {
    if (!isSubscriptionRequiredDetails(error.details)) return null
    return {
      error: {
        ...base,
        code: error.code,
        details: error.details,
      },
    }
  }

  return {
    error: {
      ...base,
      code: error.code as Exclude<MagnitudeErrorCode, UsageLimitCode | SubscriptionRequiredCode>,
    },
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
  ].some((pattern) => text.includes(pattern))
}

function classifyMagnitudeError(
  response: RejectedHttpResponse,
  parsed: ParsedMagnitudeApiError,
): ProviderRejection {
  const { error } = parsed
  const base = {
    message: error.message,
  }

  if (
    isUsageLimitCode(error.code)
    && "details" in error
    && error.details.category === "usage_limit_exceeded"
  ) {
    return {
      _tag: "UsageLimitExceeded",
      ...base,
      window: error.details.violatedWindow,
      resetAt: error.details.violatedBudget.windowEnd,
    }
  }

  switch (error.code) {
    case "subscription_required":
      return { _tag: "SubscriptionRequired", ...base }
    case "invalid_api_key":
      return { _tag: "AuthRejected", ...base }
    case "model_not_found":
      return { _tag: "ModelUnavailable", ...base }
    case "model_not_multimodal":
      return { _tag: "ModelCapabilityMissing", ...base }
    case "model_not_grammar_compatible":
      return { _tag: "ProviderCapabilityMissing", ...base }
    case "provider_rate_limited":
      return {
        _tag: "RateLimited",
        ...base,
        retryPolicy: retryPolicy(true, response.retryAfterMs),
      }
    case "internal_server_error":
    case "invariant_violation":
      return { _tag: "ProviderInvariantViolation", ...base }
    case "provider_error":
    case "upstream_unavailable":
    case "stream_interrupted":
      return {
        _tag: "UpstreamFailure",
        ...base,
        retryPolicy: retryPolicy(true, null),
      }
    default:
      if (error.type === "authentication_error" || response.status === 401 || response.status === 403) {
        return { _tag: "AuthRejected", ...base }
      }
      if (error.type === "rate_limit_error" || response.status === 429) {
        return {
          _tag: "RateLimited",
          ...base,
          retryPolicy: retryPolicy(true, response.retryAfterMs),
        }
      }
      if (isContextLimit(error.message)) {
        return { _tag: "ContextLimitExceeded", ...base }
      }
      return { _tag: "InvalidRequest", ...base }
  }
}

export function classifyMagnitudeRejectedResponse(
  call: ProviderCall,
  response: RejectedHttpResponse,
): StreamStartProviderRejection | StreamStartProviderCorrectnessViolation {
  const parsed = tryParseErrorBody(response.body)
  if (parsed === null) {
    return new StreamStartProviderCorrectnessViolation({
      call,
      response,
      violation: {
        _tag: "InvalidErrorEnvelope",
        status: response.status,
        body: payloadSample(response.body),
        issue: { message: "Magnitude error response did not match the expected envelope shape" },
      },
    })
  }

  return new StreamStartProviderRejection({
    call,
    response,
    rejection: classifyMagnitudeError(response, parsed),
  })
}
