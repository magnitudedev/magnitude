/**
 * AUTO-GENERATED from magnitude-provider/lib/api-contract.
 * Run npm run sync:types in magnitude-provider to update this file.
 */

import type {
  BillingWindowBudget,
  BillingWindowName,
  ProSubscriptionStatus,
} from "./errors"

export type UsagePeriod = "24h" | "3d" | "7d" | "14d" | "30d" | "all"

export interface CloudUsageResponse {
  readonly data: {
    readonly meta: {
      readonly generatedAt: string
      readonly autumnConfigured: boolean
    }
    readonly subscription: {
      readonly status: ProSubscriptionStatus
      readonly plan: {
        readonly id: "pro"
        readonly label: "Pro"
        readonly priceCents: number
      }
    }
    readonly usageWindows: Partial<Record<BillingWindowName, BillingWindowBudget>>
    readonly usage: {
      readonly period: UsagePeriod
      readonly rangeStart: string
      readonly rangeEnd: string
      readonly totals: {
        readonly requestCount: number
        readonly costCents: number
        readonly inputTokens: number
        readonly outputTokens: number
      }
      readonly byModel: ReadonlyArray<{
        readonly model: string
        readonly requestCount: number
        readonly costCents: number
        readonly inputTokens: number
        readonly outputTokens: number
      }>
      readonly dailyTokens: ReadonlyArray<{
        readonly date: string
        readonly inputTokens: number
        readonly outputTokens: number
        readonly topModel: string | null
      }>
    }
  }
}
