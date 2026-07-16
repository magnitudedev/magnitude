/**
 * AUTO-GENERATED from magnitude-provider/lib/api-contract.
 * Run npm run sync:types in magnitude-provider to update this file.
 */

export interface MagnitudeApiError {
  readonly error: {
    readonly message: string
    readonly type: MagnitudeErrorType
    readonly code: MagnitudeErrorCode
    readonly param: string | null
    readonly details?: MagnitudeErrorDetails
  }
}

export type MagnitudeErrorType =
  | "invalid_request_error"
  | "authentication_error"
  | "insufficient_quota"
  | "rate_limit_error"
  | "server_error"
  | "service_unavailable"

export type MagnitudeErrorCode =
  | "invalid_api_key"
  | "invalid_body"
  | "unsupported_field"
  | "unsupported_n"
  | "invalid_image_url"
  | "invalid_multimodal_role"
  | "model_not_found"
  | "model_not_multimodal"
  | "model_not_grammar_compatible"
  | "subscription_required"
  | "usage_limit_exceeded_five_hour"
  | "usage_limit_exceeded_weekly"
  | "usage_limit_exceeded_monthly"
  | "provider_rate_limited"
  | "internal_server_error"
  | "provider_error"
  | "invariant_violation"
  | "upstream_unavailable"
  | "stream_interrupted"

export type MagnitudeErrorDetails = SubscriptionRequiredDetails | UsageLimitDetails

export type BillingWindowName = "five_hour" | "weekly" | "monthly"
export type ProSubscriptionStatus = "active" | "not_subscribed"

export interface BillingWindowBudget {
  readonly limitCents: number
  readonly usedCents: number
  readonly remainingCents: number
  readonly windowStart: string
  readonly windowEnd: string
  readonly remainingMs: number
}

export interface UsageLimitDetails {
  readonly category: "usage_limit_exceeded"
  readonly violatedWindow: BillingWindowName
  readonly windows: Record<BillingWindowName, BillingWindowBudget>
  readonly violatedBudget: BillingWindowBudget
}

export interface SubscriptionRequiredDetails {
  readonly category: "subscription_required"
  readonly requiredPlanId: "pro"
}
