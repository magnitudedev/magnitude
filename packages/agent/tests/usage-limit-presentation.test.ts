import { describe, expect, test } from "vitest"
import { present } from "../src/errors/present"

describe("usage-limit presentation", () => {
  test("shows a non-retryable cloud usage message and billing link", () => {
    const result = present({
      _tag: "ProviderNotReady",
      detail: {
        _tag: "UsageLimitExceeded",
        message: "You've reached your monthly cloud usage limit.",
        window: "monthly",
        resetAt: "2026-08-01T00:00:00.000Z",
      },
      requestId: "request",
    })

    expect(result.retryable).toBe(false)
    expect(result.message).toContain("monthly cloud usage limit")
    expect(result.cta).toEqual({
      kind: "url",
      label: "View cloud usage",
      url: "https://app.magnitude.dev/billing",
    })
  })

  test("shows a Pro subscription link when cloud access is not active", () => {
    const result = present({
      _tag: "ProviderNotReady",
      detail: {
        _tag: "SubscriptionRequired",
        message: "Magnitude Pro is required to use cloud models.",
      },
      requestId: "request",
    })

    expect(result.retryable).toBe(false)
    expect(result.message).toContain("Magnitude Pro")
    expect(result.cta).toEqual({
      kind: "url",
      label: "Start Magnitude Pro",
      url: "https://app.magnitude.dev/billing",
    })
  })
})
