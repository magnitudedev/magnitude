import { Schema } from "effect"

export const ServingModelId = Schema.NonEmptyString.pipe(Schema.brand("ServingModelId"))
export type ServingModelId = typeof ServingModelId.Type
export const ServingUsageRequest = Schema.Struct({
  period: Schema.Literal("Today", "AllTime"),
  timeZone: Schema.String,
  model: Schema.optionalWith(ServingModelId, { as: "Option", exact: true }),
})
export type ServingUsageRequest = typeof ServingUsageRequest.Type
export const ServingActivityDate = Schema.String.pipe(Schema.pattern(/^\d{4}-\d{2}-\d{2}$/), Schema.brand("ServingActivityDate"))
export const ServingActivityDay = Schema.Struct({ date: ServingActivityDate, totalTokens: Schema.NonNegativeInt })
export const ServingUsageSnapshot = Schema.Union(
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
  Schema.TaggedStruct("Available", {
    since: Schema.NullOr(Schema.Number),
    dailyActivity: Schema.Array(ServingActivityDay),
    requests: Schema.NonNegativeInt,
    incompleteRequests: Schema.NonNegativeInt,
    inputTokens: Schema.NonNegativeInt,
    cachedInputTokens: Schema.NonNegativeInt,
    outputTokens: Schema.NonNegativeInt,
    totalTokens: Schema.NonNegativeInt,
    cachedInputRequests: Schema.NonNegativeInt,
    tokensPerSecond: Schema.NullOr(Schema.Number),
    timeToFirstTokenMs: Schema.NullOr(Schema.Number),
    speedSamples: Schema.NonNegativeInt,
    latencySamples: Schema.NonNegativeInt,
    models: Schema.Array(Schema.Struct({ id: ServingModelId, requests: Schema.NonNegativeInt })),
    recordingFailures: Schema.NonNegativeInt,
  }),
)
export type ServingUsageSnapshot = typeof ServingUsageSnapshot.Type
