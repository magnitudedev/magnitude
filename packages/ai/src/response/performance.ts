import { Schema } from "effect"

const FiniteNonNegative = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())

/** Provider-reported measurements for one completed model generation. */
export const GenerationPerformanceSchema = Schema.Struct({
  generatedTokens: Schema.Number.pipe(Schema.int(), Schema.nonNegative()),
  decodeDurationMs: FiniteNonNegative,
  decodeTokensPerSecond: FiniteNonNegative,
  timeToFirstTokenMs: FiniteNonNegative,
})

export type GenerationPerformance = typeof GenerationPerformanceSchema.Type
