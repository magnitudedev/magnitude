import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import { CloudUsageResponse } from "./account"

describe("cloud usage response schema", () => {
  test("accepts an account without a Pro subscription", () => {
    const value = {
      data: {
        meta: { generatedAt: "2026-07-15T00:00:00.000Z", autumnConfigured: true },
        subscription: {
          status: "not_subscribed",
          plan: { id: "pro", label: "Pro", priceCents: 2_000 },
        },
        usageWindows: {},
        usage: {
          period: "7d",
          rangeStart: "2026-07-08T00:00:00.000Z",
          rangeEnd: "2026-07-15T00:00:00.000Z",
          totals: { requestCount: 1, costCents: 25.5, inputTokens: 10, outputTokens: 5 },
          byModel: [],
          dailyTokens: [],
        },
      },
    }

    expect(Schema.decodeUnknownSync(CloudUsageResponse)(value)).toEqual(value)
  })

  test("accepts an active Pro subscription with all usage windows", () => {
    const budget = {
      limitCents: 1_000,
      usedCents: 125.5,
      remainingCents: 874.5,
      windowStart: "2026-07-15T00:00:00.000Z",
      windowEnd: "2026-07-15T05:00:00.000Z",
      remainingMs: 1_000,
    }
    const value = {
      data: {
        meta: { generatedAt: "2026-07-15T00:00:00.000Z", autumnConfigured: true },
        subscription: {
          status: "active",
          plan: { id: "pro", label: "Pro", priceCents: 2_000 },
        },
        usageWindows: {
          five_hour: budget,
          weekly: { ...budget, limitCents: 2_000, remainingCents: 1_874.5 },
          monthly: { ...budget, limitCents: 4_000, remainingCents: 3_874.5 },
        },
        usage: {
          period: "7d",
          rangeStart: "2026-07-08T00:00:00.000Z",
          rangeEnd: "2026-07-15T00:00:00.000Z",
          totals: { requestCount: 1, costCents: 125.5, inputTokens: 10, outputTokens: 5 },
          byModel: [],
          dailyTokens: [],
        },
      },
    }

    expect(Schema.decodeUnknownSync(CloudUsageResponse)(value)).toEqual(value)
  })
})
