import type { UsagePeriod } from "@magnitudedev/protocol"
export type { UsagePeriod }

export interface BalanceResponse {
  readonly data: {
    readonly meta: {
      readonly generatedAt: string
      readonly autumnConfigured: boolean
    }
    readonly balance: {
      readonly cents: number
    }
    readonly autoReload: {
      readonly enabled: boolean
      readonly thresholdCents: number
      readonly amountCents: number
      readonly lastFailure: {
        readonly reason: string
        readonly at: string | null
      } | null
    }
    readonly hasPaymentMethod: boolean
    readonly recentTopups: ReadonlyArray<{
      readonly at: string | null
      readonly chargedCents: number
      readonly invoiceUrl: string | null
      readonly status: string | null
    }>
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
