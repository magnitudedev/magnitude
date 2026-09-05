import { Either, Schema } from "effect"
import { MagnitudeProgressSchema, MagnitudeTimingsSchema, type MagnitudeProgress, type MagnitudeTimings } from "@magnitudedev/sdk"
export { MagnitudeProgressSchema, MagnitudeTimingsSchema }
export type { MagnitudeProgress, MagnitudeTimings }

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
