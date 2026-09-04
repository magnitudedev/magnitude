import { Either, Schema } from "effect"

const NonNegativeInteger = Schema.Number.pipe(Schema.int(), Schema.nonNegative())
const FiniteNonNegative = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())

export const MagnitudeProgressSchema = Schema.Union(
  Schema.Struct({ phase: Schema.Literal("model_loading"), fraction: Schema.Number.pipe(Schema.finite()) }),
  Schema.Struct({ phase: Schema.Literal("queued") }),
  Schema.Struct({ phase: Schema.Literal("preparing") }),
  Schema.Struct({
    phase: Schema.Literal("prefill"),
    completed_tokens: NonNegativeInteger,
    total_tokens: NonNegativeInteger,
    cached_tokens: NonNegativeInteger,
  }),
  Schema.Struct({ phase: Schema.Literal("generating") }),
)
export type MagnitudeProgress = typeof MagnitudeProgressSchema.Type

export const MagnitudeTimingsSchema = Schema.Struct({
  prompt_ms: FiniteNonNegative,
  time_to_first_token_ms: FiniteNonNegative,
  predicted_n: NonNegativeInteger,
  predicted_ms: FiniteNonNegative,
  predicted_per_second: FiniteNonNegative,
})
export type MagnitudeTimings = typeof MagnitudeTimingsSchema.Type

const ProgressChunkSchema = Schema.Struct({ progress: MagnitudeProgressSchema })
const TimingsChunkSchema = Schema.Struct({ timings: MagnitudeTimingsSchema })

export interface MagnitudeObservation {
  readonly progress?: MagnitudeProgress
  readonly timings?: MagnitudeTimings
}

export const decodeMagnitudeObservation = (value: unknown): MagnitudeObservation | undefined => {
  const progress = Schema.decodeUnknownEither(ProgressChunkSchema)(value)
  const timings = Schema.decodeUnknownEither(TimingsChunkSchema)(value)
  if (Either.isLeft(progress) && Either.isLeft(timings)) return undefined
  return {
    ...(Either.isRight(progress) ? { progress: progress.right.progress } : {}),
    ...(Either.isRight(timings) ? { timings: timings.right.timings } : {}),
  }
}
