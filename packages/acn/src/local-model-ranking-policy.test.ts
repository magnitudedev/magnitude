import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import {
  LocalModelRankingSampleMissing,
  modelRankingScores,
  normalizedModelSpeedScore,
  type LocalModelRankingCandidate,
} from "./local-model-ranking-policy"

const candidate = (input: {
  readonly intelligence?: number
  readonly quality?: number
  readonly context?: number
  readonly speed?: number
  readonly includeComparisonSample?: boolean
} = {}): LocalModelRankingCandidate => {
  const contextLength = input.context ?? 100_000
  const comparisonContext = Math.min(50_000, contextLength)
  return {
    model: {
      modelId: "model",
      qualityScore: input.intelligence ?? 80,
      fidelityRank: input.quality ?? 60,
    },
    profile: { contextLength },
    assessment: {
      performance: input.includeComparisonSample === false ? [] : [{
        contextTokens: comparisonContext,
        estimatedTokensPerSecond: input.speed ?? 40,
      }],
    },
  } as unknown as LocalModelRankingCandidate
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
  it("normalizes intelligence, target speed, and quantization quality independently", async () => {
    const scores = await Effect.runPromise(modelRankingScores(candidate({
      intelligence: 75,
      quality: 50,
      speed: 40,
    })))
    expect(scores).toEqual({
      intelligence: 0.75,
      speed: normalizedModelSpeedScore(40),
      quality: 0.5,
    })
  })

  it("clamps catalog scores to the normalized range", async () => {
    const scores = await Effect.runPromise(modelRankingScores(candidate({
      intelligence: 150,
      quality: -10,
    })))
    expect(scores.intelligence).toBe(1)
    expect(scores.quality).toBe(0)
  })

  it("fails explicitly when the bounded comparison sample is absent", async () => {
    const failure = await Effect.runPromise(Effect.flip(modelRankingScores(candidate({
      includeComparisonSample: false,
    }))))
    expect(failure).toBeInstanceOf(LocalModelRankingSampleMissing)
    expect(failure.contextTokens).toBe(50_000)
  })

  it("uses the configured context as the comparison point below 50K", async () => {
    const scores = await Effect.runPromise(modelRankingScores(candidate({
      context: 32_768,
      speed: 24,
    })))
    expect(scores.speed).toBeCloseTo(normalizedModelSpeedScore(24))
  })
})
