import { Schema } from "effect"
import { ChatCompletionProgress, Timings } from "@magnitudedev/icn-protocol/schemas"

export const MagnitudeProgressSchema = ChatCompletionProgress
export type MagnitudeProgress = typeof MagnitudeProgressSchema.Type
/** The timing fields used by companion UIs, derived from the inference wire schema. */
export const MagnitudeTimingsSchema = Timings.pipe(Schema.pick(
  "prompt_ms", "time_to_first_token_ms", "predicted_n", "predicted_ms", "predicted_per_second",
))
export type MagnitudeTimings = typeof MagnitudeTimingsSchema.Type
