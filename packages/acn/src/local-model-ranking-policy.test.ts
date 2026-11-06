import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  modelRankingScores,
  normalizedModelSpeedScore,
  type LocalModelRankingCandidate,
} from "./local-model-ranking-policy"

const candidate = (input: {
  readonly intelligence?: number
  readonly fidelity?: number
  readonly context?: number
  readonly speed?: number
  readonly includeComparisonSample?: boolean
} = {}): LocalModelRankingCandidate => {
  const contextLength = input.context ?? 100_000
  const comparisonContext = Math.min(50_000, contextLength)
  return {
    intelligenceScore: input.intelligence ?? 80,
    fidelityRank: input.fidelity ?? 60,
    profile: { contextLength },
    performance: input.includeComparisonSample === false
      ? { _tag: "Pending", measurement: { _tag: "Measuring" } }
      : {
          _tag: "Estimated",
          samples: [{
            contextTokens: comparisonContext,
            estimatedTokensPerSecond: input.speed ?? 40,
          }],
        },
  }
}

describe("normalizedModelSpeedScore", () => {
  it("is monotonic and bounded at the comparison ceiling", () => {
    const speeds = [0, 10, 40, 60, 100, 150].map(normalizedModelSpeedScore)
    expect(speeds[0]).toBe(0)
    expect(speeds.at(-2)).toBeCloseTo(1)
    expect(speeds.at(-1)).toBeCloseTo(1)
    expect(speeds).toEqual([...speeds].sort((left, right) => left - right))
  })
})

describe("modelRankingScores", () => {
  it("normalizes intelligence, target speed, and quantization fidelity independently", () => {
    const scores = modelRankingScores(candidate({
      intelligence: 75,
      fidelity: 50,
      speed: 40,
    }))
    expect(scores).toEqual({
      intelligence: 0.75,
      speed: Option.some(normalizedModelSpeedScore(40)),
      fidelity: 0.5,
    })
  })

  it("clamps catalog scores to the normalized range", () => {
    const scores = modelRankingScores(candidate({
      intelligence: 150,
      fidelity: -10,
    }))
    expect(scores.intelligence).toBe(1)
    expect(scores.fidelity).toBe(0)
  })

  it("keeps the catalog terms and omits only speed while speed is pending", () => {
    const scores = modelRankingScores(candidate({
      intelligence: 75,
      fidelity: 50,
      includeComparisonSample: false,
    }))
    expect(scores).toEqual({ intelligence: 0.75, speed: Option.none(), fidelity: 0.5 })
  })

  it("uses the configured context as the comparison point below 50K", () => {
    const scores = modelRankingScores(candidate({
      context: 32_768,
      speed: 24,
    }))
    expect(Option.getOrThrow(scores.speed)).toBeCloseTo(normalizedModelSpeedScore(24))
  })
})
