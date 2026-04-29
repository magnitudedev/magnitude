import { Data } from "effect"
import type {
  MagnitudeApiError,
  UsageLimitDetails,
  SubscriptionRequiredDetails,
} from "./contract"
import {
  defaultClassifyConnectionError,
  type ConnectionError,
  type HttpConnectionFailure,
} from "@magnitudedev/ai"

// --- Magnitude-specific errors ---

export class SubscriptionRequired extends Data.TaggedError("SubscriptionRequired")<{
  readonly message: string
  readonly details: SubscriptionRequiredDetails
}> {}

export class TrialExpired extends Data.TaggedError("TrialExpired")<{
  readonly message: string
}> {}

export class MagnitudeUsageLimitExceeded extends Data.TaggedError("MagnitudeUsageLimitExceeded")<{
  readonly message: string
  readonly details: UsageLimitDetails
}> {}

export class ModelNotGrammarCompatible extends Data.TaggedError("ModelNotGrammarCompatible")<{
  readonly message: string
  readonly model: string
}> {}

export class RoleNotFound extends Data.TaggedError("RoleNotFound")<{
  readonly message: string
  readonly role: string
}> {}

export type MagnitudeConnectionError =
  | ConnectionError
  | SubscriptionRequired
  | TrialExpired
  | MagnitudeUsageLimitExceeded
  | ModelNotGrammarCompatible
  | RoleNotFound

// --- Body parser ---

export function tryParseErrorBody(body: string): MagnitudeApiError | null {
  try {
    const parsed = JSON.parse(body)
    if (parsed?.error?.type && parsed?.error?.code) return parsed as MagnitudeApiError
    return null
  } catch {
    return null
  }
}

// --- Classifier ---

export function classifyMagnitudeConnectionError(
  sourceId: string,
  failure: HttpConnectionFailure,
): MagnitudeConnectionError {
  const parsed = tryParseErrorBody(failure.body)

  if (parsed) {
    switch (parsed.error.code) {
      case "subscription_required":
        return new SubscriptionRequired({
          message: parsed.error.message,
          details: parsed.error.details as SubscriptionRequiredDetails,
        })
      case "trial_expired":
        return new TrialExpired({ message: parsed.error.message })
      case "usage_limit_exceeded_five_hour":
      case "usage_limit_exceeded_weekly":
      case "usage_limit_exceeded_monthly":
        return new MagnitudeUsageLimitExceeded({
          message: parsed.error.message,
          details: parsed.error.details as UsageLimitDetails,
        })
      case "model_not_grammar_compatible":
        return new ModelNotGrammarCompatible({
          message: parsed.error.message,
          model: sourceId,
        })
      case "role_not_found":
        return new RoleNotFound({
          message: parsed.error.message,
          role: sourceId,
        })
    }
  }

  return defaultClassifyConnectionError(sourceId, failure)
}
