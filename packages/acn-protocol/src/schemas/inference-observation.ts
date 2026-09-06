import { Schema } from "effect"
import { ChatCompletionProgress, Timings } from "@magnitudedev/icn-protocol/schemas"

export const InferenceObservationIdSchema = Schema.UUID.pipe(Schema.brand("InferenceObservationId"))
export type InferenceObservationId = typeof InferenceObservationIdSchema.Type
export const InferenceObservationGroupIdSchema = Schema.UUID.pipe(Schema.brand("InferenceObservationGroupId"))
export type InferenceObservationGroupId = typeof InferenceObservationGroupIdSchema.Type

// Pick from the canonical variants: no arbitrary extension fields or recursive
// JSON payloads cross the observation boundary.
export const InferenceProgressSchema = Schema.Union(
  ChatCompletionProgress.members[0].pipe(Schema.pick("phase", "fraction")),
  ChatCompletionProgress.members[1].pipe(Schema.pick("phase")),
  ChatCompletionProgress.members[2].pipe(Schema.pick("phase")),
  ChatCompletionProgress.members[3].pipe(Schema.pick("phase", "cached_tokens", "completed_tokens", "total_tokens")),
  ChatCompletionProgress.members[4].pipe(Schema.pick("phase")),
)
export const InferenceTimingsSchema = Timings.pipe(Schema.pick(
  "prompt_ms", "time_to_first_token_ms", "predicted_n", "predicted_ms", "predicted_per_second",
))

export const InferenceObservationSchema = Schema.Struct({
  requestId: InferenceObservationIdSchema,
  state: Schema.Literal("Active", "Ended"),
  elapsedMs: Schema.NonNegative,
  progress: Schema.optionalWith(InferenceProgressSchema, { as: "Option", exact: true }),
  timings: Schema.optionalWith(InferenceTimingsSchema, { as: "Option", exact: true }),
})
export type InferenceObservation = typeof InferenceObservationSchema.Type
export const InferenceObservationsSchema = Schema.Array(InferenceObservationSchema).pipe(Schema.maxItems(128))
