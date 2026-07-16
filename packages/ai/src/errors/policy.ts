import { Context, Data, Effect, Layer, Option } from "effect"
import {
  StreamStartFailure,
  StreamFailure,
  ModelStreamTerminal,
  ProviderRejection,
} from "./failure"

// ---------------------------------------------------------------------------
// RetryAfter ADT
// ---------------------------------------------------------------------------

export type RetryAfter = Data.TaggedEnum<{
  RetryAfterMs: { readonly ms: number }
  NoRetryAfter: {}
}>

const retryAfterEnum = Data.taggedEnum<RetryAfter>()

export const RetryAfter = {
  RetryAfterMs: (args: { readonly ms: number }): Extract<RetryAfter, { readonly _tag: "RetryAfterMs" }> => ({
    _tag: "RetryAfterMs",
    ...args,
  }),
  NoRetryAfter: (): Extract<RetryAfter, { readonly _tag: "NoRetryAfter" }> => ({
    _tag: "NoRetryAfter",
  }),
  $is: retryAfterEnum.$is,
  $match: retryAfterEnum.$match,
}

// ---------------------------------------------------------------------------
// UpstreamRetryability ADT
// ---------------------------------------------------------------------------

export type UpstreamRetryability = Data.TaggedEnum<{
  UpstreamRetryable: { readonly retryAfter: RetryAfter }
  UpstreamNotRetryable: { readonly reason: NonRetryableReason }
}>

const upstreamRetryabilityEnum = Data.taggedEnum<UpstreamRetryability>()

export const UpstreamRetryability = {
  UpstreamRetryable: (
    args: { readonly retryAfter: RetryAfter },
  ): Extract<UpstreamRetryability, { readonly _tag: "UpstreamRetryable" }> => ({
    _tag: "UpstreamRetryable",
    ...args,
  }),
  UpstreamNotRetryable: (
    args: { readonly reason: NonRetryableReason },
  ): Extract<UpstreamRetryability, { readonly _tag: "UpstreamNotRetryable" }> => ({
    _tag: "UpstreamNotRetryable",
    ...args,
  }),
  $is: upstreamRetryabilityEnum.$is,
  $match: upstreamRetryabilityEnum.$match,
}

type NonRetryableReason =
  | "auth"
  | "invalid_request"
  | "context_limit"
  | "billing"
  | "malformed_provider_data"
  | "internal_defect"
  | "model_unavailable"
  | "non_retryable_terminal"
  | "provider_error_not_retryable"

// ---------------------------------------------------------------------------
// BillingDisposition ADT
// ---------------------------------------------------------------------------

export type BillingDisposition = Data.TaggedEnum<{
  UseProviderUsage: {}
  UseLocalEstimate: { readonly reason: string }
  NoCharge: { readonly reason: string }
  FailClosed: { readonly reason: string }
}>

const billingDispositionEnum = Data.taggedEnum<BillingDisposition>()

export const BillingDisposition = {
  UseProviderUsage: (): Extract<BillingDisposition, { readonly _tag: "UseProviderUsage" }> => ({
    _tag: "UseProviderUsage",
  }),
  UseLocalEstimate: (
    args: { readonly reason: string },
  ): Extract<BillingDisposition, { readonly _tag: "UseLocalEstimate" }> => ({
    _tag: "UseLocalEstimate",
    ...args,
  }),
  NoCharge: (
    args: { readonly reason: string },
  ): Extract<BillingDisposition, { readonly _tag: "NoCharge" }> => ({
    _tag: "NoCharge",
    ...args,
  }),
  FailClosed: (
    args: { readonly reason: string },
  ): Extract<BillingDisposition, { readonly _tag: "FailClosed" }> => ({
    _tag: "FailClosed",
    ...args,
  }),
  $is: billingDispositionEnum.$is,
  $match: billingDispositionEnum.$match,
}

// ---------------------------------------------------------------------------
// AiRetryPolicy service
// ---------------------------------------------------------------------------

export interface AiRetryPolicyService {
  /**
   * Should the upstream provider call be retried? Depends on provider-specific
   * rules, model configuration, and route configuration.
   */
  upstreamRetryability: (
    failure: StreamStartFailure | StreamFailure | ModelStreamTerminal,
  ) => Effect.Effect<UpstreamRetryability, never, AiRetryPolicy>
}

export class AiRetryPolicy extends Context.Tag("AiRetryPolicy")<
  AiRetryPolicy,
  AiRetryPolicyService
>() {}

// ---------------------------------------------------------------------------
// AiBillingPolicy service
// ---------------------------------------------------------------------------

export interface AiBillingPolicyService {
  billingDisposition: (
    terminal: ModelStreamTerminal,
  ) => Effect.Effect<BillingDisposition, never, AiBillingPolicy>
}

export class AiBillingPolicy extends Context.Tag("AiBillingPolicy")<
  AiBillingPolicy,
  AiBillingPolicyService
>() {}

// ---------------------------------------------------------------------------
// Default implementation — pure structural rules, no provider magic
// ---------------------------------------------------------------------------

function retryabilityForTerminal(
  terminal: ModelStreamTerminal,
): UpstreamRetryability {
  switch (terminal._tag) {
    case "StreamCompleted":
      return UpstreamRetryability.UpstreamNotRetryable({ reason: "non_retryable_terminal" })
    case "StreamFailed":
      return retryabilityForStreamFailure(terminal.cause)
  }
}

function retryabilityForStreamFailure(
  failure: StreamStartFailure | StreamFailure,
): UpstreamRetryability {
  switch (failure._tag) {
    case "StreamStartOperationalFailure":
    case "StreamOperationalFailure":
      return UpstreamRetryability.UpstreamRetryable({
        retryAfter: RetryAfter.NoRetryAfter(),
      })

    case "StreamStartProviderRejection":
      return providerRejectionRetryable(failure.rejection)

    case "StreamProviderError":
      return providerErrorRetryable(failure.providerError, failure.response.status, null)

    case "StreamStartProviderCorrectnessViolation":
    case "StreamProviderCorrectnessViolation":
      return UpstreamRetryability.UpstreamNotRetryable({
        reason: "malformed_provider_data",
      })

    case "StreamStartClientCorrectnessViolation":
    case "StreamClientCorrectnessViolation":
      return UpstreamRetryability.UpstreamNotRetryable({
        reason: "internal_defect",
      })
  }
}

function providerErrorRetryable(
  error: {
    readonly message: string
    readonly type: string | null
    readonly code: string | null
    readonly param: string | null
  },
  status: number,
  retryAfterMs: number | null,
): UpstreamRetryability {
  const text = [error.message, error.type ?? "", error.code ?? "", error.param ?? ""]
    .join(" ")
    .toLowerCase()
  if ([
    "prompt is too long",
    "token count exceeds the maximum",
    "maximum context length",
    "context_length_exceeded",
    "exceeded model token limit",
  ].some((p) => text.includes(p))) {
    return UpstreamRetryability.UpstreamNotRetryable({ reason: "context_limit" })
  }
  if (status === 401 || status === 403 || [
    "missing_scope",
    "insufficient_scope",
    "invalid_token",
    "token expired",
    "expired token",
    "unauthorized",
    "forbidden",
    "authentication",
  ].some((p) => text.includes(p))) {
    return UpstreamRetryability.UpstreamNotRetryable({ reason: "auth" })
  }
  if (status === 429) {
    return retryAfterMs !== null
      ? UpstreamRetryability.UpstreamRetryable({ retryAfter: RetryAfter.RetryAfterMs({ ms: retryAfterMs }) })
      : UpstreamRetryability.UpstreamRetryable({ retryAfter: RetryAfter.NoRetryAfter() })
  }
  if (status >= 500) {
    return UpstreamRetryability.UpstreamRetryable({ retryAfter: RetryAfter.NoRetryAfter() })
  }
  if (status >= 400 && status < 500) {
    return UpstreamRetryability.UpstreamNotRetryable({ reason: "invalid_request" })
  }
  const retryable = [
    "timeout",
    "timed_out",
    "temporarily_unavailable",
    "unavailable",
    "overloaded",
    "rate_limit",
    "server_error",
    "internal_error",
    "upstream_unavailable",
    "stream_interrupted",
  ].some((p) => text.includes(p))
  return retryable
    ? UpstreamRetryability.UpstreamRetryable({
        retryAfter: RetryAfter.NoRetryAfter(),
      })
    : UpstreamRetryability.UpstreamNotRetryable({
        reason: "provider_error_not_retryable",
      })
}

/**
 * Pure default retryability decision for a given failure or terminal.
 * Use this directly when you need a synchronous decision, or use the
 * {@link AiRetryPolicy} service when you want to allow override.
 */
export function defaultRetryabilityForFailure(
  failure: StreamStartFailure | StreamFailure | ModelStreamTerminal,
): UpstreamRetryability {
  switch (failure._tag) {
    case "StreamCompleted":
    case "StreamFailed":
      return retryabilityForTerminal(failure)
    default:
      return retryabilityForStreamFailure(failure)
  }
}

function providerRejectionRetryable(rejection: ProviderRejection): UpstreamRetryability {
  if ("retryPolicy" in rejection) {
    return rejection.retryPolicy.retry
      ? UpstreamRetryability.UpstreamRetryable({
          retryAfter: Option.isSome(rejection.retryPolicy.retryAfterMs)
            ? RetryAfter.RetryAfterMs({ ms: rejection.retryPolicy.retryAfterMs.value })
            : RetryAfter.NoRetryAfter(),
        })
      : UpstreamRetryability.UpstreamNotRetryable({ reason: "provider_error_not_retryable" })
  }

  return UpstreamRetryability.UpstreamNotRetryable({ reason: nonRetryableReasonForTag(rejection._tag) })
}

function nonRetryableReasonForTag(tag: ProviderRejection["_tag"]): NonRetryableReason {
  switch (tag) {
    case "AuthRejected":
      return "auth"
    case "SubscriptionRequired":
    case "UsageLimitExceeded":
      return "billing"
    case "ModelUnavailable":
    case "ModelCapabilityMissing":
    case "ProviderCapabilityMissing":
      return "model_unavailable"
    case "ContextLimitExceeded":
      return "context_limit"
    case "InvalidRequest":
      return "invalid_request"
    case "ProviderInvariantViolation":
      return "provider_error_not_retryable"
    default:
      return "provider_error_not_retryable"
  }
}

const defaultRetryPolicy: AiRetryPolicyService = {
  upstreamRetryability: (failure) =>
    Effect.succeed(defaultRetryabilityForFailure(failure)),
}

export const AiRetryPolicyLive = Layer.succeed(AiRetryPolicy, defaultRetryPolicy)

const defaultBillingPolicy: AiBillingPolicyService = {
  billingDisposition: (terminal) =>
    Effect.succeed(
      terminal._tag === "StreamCompleted" && terminal.usage._tag === "UsageReported"
        ? BillingDisposition.UseProviderUsage()
        : BillingDisposition.UseLocalEstimate({ reason: "terminal_not_completed_or_usage_missing" })
    ),
}

export const AiBillingPolicyLive = Layer.succeed(AiBillingPolicy, defaultBillingPolicy)
