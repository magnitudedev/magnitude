import { Schema } from "effect"

export const UsagePeriod = Schema.Literal("24h", "3d", "7d", "14d", "30d", "all")
export type UsagePeriod = Schema.Schema.Type<typeof UsagePeriod>

const UsageTotals = Schema.Struct({
  requestCount: Schema.Number,
  costCents: Schema.Number,
  inputTokens: Schema.Number,
  outputTokens: Schema.Number,
})

export const CloudUsageResponse = Schema.Struct({
  data: Schema.Struct({
    meta: Schema.Struct({
      generatedAt: Schema.String,
      autumnConfigured: Schema.Boolean,
    }),
    subscription: Schema.Struct({
      status: Schema.Literal("active", "not_subscribed"),
      plan: Schema.Struct({
        id: Schema.Literal("pro"),
        label: Schema.Literal("Pro"),
        priceCents: Schema.Number,
      }),
    }),
    usageWindows: Schema.partial(Schema.Record({
      key: Schema.Literal("five_hour", "weekly", "monthly"),
      value: Schema.Struct({
        limitCents: Schema.Number,
        usedCents: Schema.Number,
        remainingCents: Schema.Number,
        windowStart: Schema.String,
        windowEnd: Schema.String,
        remainingMs: Schema.Number,
      }),
    })),
    usage: Schema.Struct({
      period: UsagePeriod,
      rangeStart: Schema.String,
      rangeEnd: Schema.String,
      totals: UsageTotals,
      byModel: Schema.Array(Schema.Struct({
        model: Schema.String,
        requestCount: Schema.Number,
        costCents: Schema.Number,
        inputTokens: Schema.Number,
        outputTokens: Schema.Number,
      })),
      dailyTokens: Schema.Array(Schema.Struct({
        date: Schema.String,
        inputTokens: Schema.Number,
        outputTokens: Schema.Number,
        topModel: Schema.NullOr(Schema.String),
      })),
    }),
  }),
})
export type CloudUsageResponse = Schema.Schema.Type<typeof CloudUsageResponse>
