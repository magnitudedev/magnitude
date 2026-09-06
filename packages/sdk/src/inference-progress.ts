import { InferenceProgressSchema, InferenceTimingsSchema } from "@magnitudedev/acn-protocol"

export const MagnitudeProgressSchema = InferenceProgressSchema
export type MagnitudeProgress = typeof MagnitudeProgressSchema.Type
/** The timing fields used by companion UIs, derived from the inference wire schema. */
export const MagnitudeTimingsSchema = InferenceTimingsSchema
export type MagnitudeTimings = typeof MagnitudeTimingsSchema.Type
export { InferenceObservationsSchema } from "@magnitudedev/acn-protocol"
