import { describe, expect, test } from "vitest"
import { rejectedHttpResponse, type ProviderCall } from "@magnitudedev/ai"
import { classifyMagnitudeRejectedResponse, tryParseErrorBody } from "./errors"

const budget = {
  limitCents: 500,
  usedCents: 505.25,
  remainingCents: 0,
  windowStart: "2026-07-01T00:00:00.000Z",
  windowEnd: "2026-07-31T00:00:00.000Z",
  remainingMs: 10_000,
}

const call: ProviderCall = {
  provider: "magnitude",
  model: "glm-5.2",
  method: "POST",
  url: "https://api.magnitude.dev/v1/chat/completions",
}

describe("Magnitude usage-limit errors", () => {
  test("parses a Pro monthly usage limit", () => {
    const parsed = tryParseErrorBody(JSON.stringify({
      error: {
        message: "You've reached your monthly cloud usage limit.",
        type: "rate_limit_error",
        code: "usage_limit_exceeded_monthly",
        param: null,
        details: {
          category: "usage_limit_exceeded",
          violatedWindow: "monthly",
          windows: {
            five_hour: budget,
            weekly: budget,
            monthly: budget,
          },
          violatedBudget: budget,
        },
      },
    }))

    expect(parsed?.error.code).toBe("usage_limit_exceeded_monthly")
    expect(parsed && "details" in parsed.error ? parsed.error.details.category : null)
      .toBe("usage_limit_exceeded")
  })

  test("parses a subscription-required response", () => {
    const body = JSON.stringify({
      error: {
        message: "Magnitude Pro is required to use cloud models.",
        type: "insufficient_quota",
        code: "subscription_required",
        param: null,
        details: {
          category: "subscription_required",
          requiredPlanId: "pro",
        },
      },
    })
    const parsed = tryParseErrorBody(body)

    expect(parsed?.error.code).toBe("subscription_required")

    const classified = classifyMagnitudeRejectedResponse(
      call,
      rejectedHttpResponse(402, new Headers(), body),
    )
    expect(classified._tag).toBe("StreamStartProviderRejection")
    if (classified._tag !== "StreamStartProviderRejection") return
    expect(classified.rejection).toEqual({
      _tag: "SubscriptionRequired",
      message: "Magnitude Pro is required to use cloud models.",
    })
  })

  test("rejects malformed usage-limit details instead of retrying a vague 429", () => {
    expect(tryParseErrorBody(JSON.stringify({
      error: {
        message: "limit",
        type: "rate_limit_error",
        code: "usage_limit_exceeded_weekly",
        param: null,
        details: { category: "usage_limit_exceeded" },
      },
    }))).toBeNull()
  })
})
