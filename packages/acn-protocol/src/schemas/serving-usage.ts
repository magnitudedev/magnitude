import { Schema } from "effect"

export const ServingModelId = Schema.NonEmptyString.pipe(Schema.brand("ServingModelId"))
export const ServingUsageRequest = Schema.Struct({
  period: Schema.Literal("Today", "AllTime"),
  timeZone: Schema.String,
  model: Schema.optionalWith(ServingModelId, { as: "Option", exact: true }),
})
export type ServingUsageRequest = typeof ServingUsageRequest.Type
const Count = Schema.Number.pipe(Schema.int(), Schema.nonNegative())
export const ServingActivityDate = Schema.String.pipe(Schema.pattern(/^\d{4}-\d{2}-\d{2}$/), Schema.brand("ServingActivityDate"))
export const ServingActivityDay = Schema.Struct({ date: ServingActivityDate, totalTokens: Count })
export const ServingUsageSnapshot = Schema.Union(
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
  Schema.TaggedStruct("Available", {
    since: Schema.NullOr(Schema.Number),
    dailyActivity: Schema.Array(ServingActivityDay),
    requests: Count,
    incompleteRequests: Count,
    inputTokens: Count,
    cachedInputTokens: Count,
    outputTokens: Count,
    totalTokens: Count,
    cachedInputRequests: Count,
    tokensPerSecond: Schema.NullOr(Schema.Number),
    timeToFirstTokenMs: Schema.NullOr(Schema.Number),
    speedSamples: Count,
    latencySamples: Count,
    models: Schema.Array(Schema.Struct({ id: ServingModelId, requests: Count })),
    recordingFailures: Count,
  }),
)
export type ServingUsageSnapshot = typeof ServingUsageSnapshot.Type
