import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  LocalInferenceMemoryDomainIdSchema,
  ModelFileIdSchema,
  ModelOfferingTargetIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  OfferingAssessmentIdSchema,
  RecommendableModelIdSchema,
  type Recommendation,
} from "@magnitudedev/protocol"
import {
  MINIMUM_EXPECTED_TOKENS_PER_SECOND,
  conservativeGenerationSpeed,
  selectRecommendationPortfolio,
  type RecommendationCandidate,
} from "./local-model-recommendation-policy"

const GIB = 1024 ** 3

const candidate = (input: {
  readonly id: string
  readonly checkpoint?: string
  readonly artifact?: string
  readonly score?: number
  readonly provenance?: string
  readonly fidelity?: number
  readonly context?: 100_000 | 200_000
  readonly expected?: number
  readonly lower?: number
  readonly upper?: number
  readonly confidence?: "high" | "moderate" | "low"
  readonly runtimeGiB?: number
  readonly downloadGiB?: number
  readonly capacityGiB?: number
  readonly architecture?: "dense" | "moe"
}): RecommendationCandidate => {
  const checkpointId = input.checkpoint ?? input.id
  const artifactId = input.artifact ?? `${checkpointId}:q${input.fidelity ?? 60}`
  const context = input.context ?? 200_000
  const expected = input.expected ?? 30
  const fidelity = input.fidelity ?? 60
  const runtimeBytes = (input.runtimeGiB ?? 24) * GIB
  const downloadBytes = (input.downloadGiB ?? input.runtimeGiB ?? 20) * GIB
  const capacityBytes = (input.capacityGiB ?? 64) * GIB
  const packageId = ModelPackageIdSchema.make(`package_${input.id}`)
  const profile = { contextLength: context, parallelSequences: 1 }
  const configurationId = ModelServingConfigurationIdSchema.make(`${input.id}:ctx${context}`)
  return {
    model: {
      id: RecommendableModelIdSchema.make(artifactId),
      checkpointId,
      targetId: ModelOfferingTargetIdSchema.make(`target_${input.id}`),
      target: {
        _tag: "Package",
        package: {
          id: packageId,
          source: {
            _tag: "HuggingFace",
            repository: "owner/repo",
            revision: "commit",
          },
          files: [{
            id: ModelFileIdSchema.make(`file_${input.id}`),
            path: `${input.id}.gguf`,
            role: "weights",
            sizeBytes: downloadBytes,
            sha256: "a".repeat(64),
          }],
          relationships: [],
          properties: {
            format: "gguf",
            quantization: `Q${fidelity}`,
            architecture: input.architecture ?? "dense",
            maximumContextLength: context,
          },
        },
      },
      eligibleServingProfiles: [profile],
      displayName: input.id,
      description: "Test fixture",
      license: "test",
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: {
          supported: false,
          efforts: [],
          defaultEffort: Option.none(),
        },
      },
      qualityScore: input.score ?? 0,
      qualityScoreProvenance: input.provenance ?? "measured_terminal_bench_2.1",
      fidelityRank: fidelity,
      quantizationAware: false,
      qualityEvidence: ["Test evidence"],
    },
    profile,
    assessment: {
      _tag: "Fits",
      profile,
      configurationId,
      assessmentId: OfferingAssessmentIdSchema.make(`assessment_${input.id}_${context}`),
      memory: [{
        memoryDomainId: LocalInferenceMemoryDomainIdSchema.make("memory"),
        capacityBytes,
        requiredBytes: runtimeBytes,
        requiredReserveBytes: 0,
        remainingBytes: capacityBytes - runtimeBytes,
      }],
      performance: Option.some({
        contextTokens: context,
        lowerTokensPerSecond: input.lower ?? expected * 0.85,
        estimatedTokensPerSecond: expected,
        upperTokensPerSecond: input.upper ?? expected * 1.15,
        confidence: input.confidence ?? "high",
        method: "test-estimator",
      }),
      performanceUnavailable: Option.none(),
    },
    artifactId,
    checkpointId,
    capability: input.score === undefined
      ? undefined
      : {
          score: input.score,
          provenance: input.provenance ?? "measured_terminal_bench_2.1",
        },
    fidelityRank: fidelity,
    quantizationAware: false,
    estimatedRuntimeBytes: runtimeBytes,
    stableCapacityBudgetBytes: capacityBytes,
    totalDownloadBytes: downloadBytes,
  }
}

const byIntent = (
  recommendations: readonly Recommendation[],
  intent: Recommendation["intent"],
): Recommendation | undefined =>
  recommendations.find((recommendation) => recommendation.intent === intent)

describe("local model multicriteria recommendation policy", () => {
  it("prefers 200K for an artifact when it clears the responsiveness floor", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "model-100", checkpoint: "model", artifact: "model:q6", score: 50, context: 100_000, expected: 40 }),
      candidate({ id: "model-200", checkpoint: "model", artifact: "model:q6", score: 50, context: 200_000, expected: 20 }),
    ])
    expect(byIntent(recommendations, "balanced")?.configuration.profile.contextLength).toBe(200_000)
  })

  it("falls back to 100K when the 200K profile misses the speed floor", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "model-100", checkpoint: "model", artifact: "model:q6", score: 50, context: 100_000, expected: 30 }),
      candidate({ id: "model-200", checkpoint: "model", artifact: "model:q6", score: 50, context: 200_000, expected: 9 }),
    ])
    expect(byIntent(recommendations, "balanced")?.configuration.profile.contextLength).toBe(100_000)
  })

  it("excludes missing or sub-floor speed evidence from every intent", () => {
    const slow = candidate({
      id: "slow",
      score: 90,
      expected: MINIMUM_EXPECTED_TOKENS_PER_SECOND - 0.1,
    })
    const missingBase = candidate({ id: "missing", score: 80 })
    const missing = {
      ...missingBase,
      assessment: { ...missingBase.assessment, performance: Option.none() },
    }
    expect(selectRecommendationPortfolio([slow, missing])).toEqual([])
  })

  it("applies the floor at the same one-decimal precision shown to users", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({
        id: "rounded-baseline",
        expected: MINIMUM_EXPECTED_TOKENS_PER_SECOND - 0.049,
      }),
    ])

    expect(byIntent(recommendations, "balanced")?.displayName).toBe("rounded-baseline")
  })

  it("builds a useful 64 GiB-class portfolio with a usable dense quality option", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "qwen27", score: 60.7, fidelity: 40, expected: 11.7, context: 100_000, runtimeGiB: 35, downloadGiB: 33 }),
      candidate({ id: "qwen35-q6", checkpoint: "qwen35", artifact: "qwen35:q6", score: 44.9, fidelity: 60, expected: 35.2, runtimeGiB: 34, downloadGiB: 29.7, architecture: "moe" }),
      candidate({ id: "qwen35-q8", checkpoint: "qwen35", artifact: "qwen35:q8", score: 44.9, fidelity: 80, expected: 34.1, runtimeGiB: 41, downloadGiB: 35.8, architecture: "moe" }),
      candidate({ id: "gemma26-100", checkpoint: "gemma26", score: 39, fidelity: 58, expected: 42, context: 100_000, runtimeGiB: 40, downloadGiB: 13.3, architecture: "moe" }),
      candidate({ id: "qwen4", score: 25.8, fidelity: 40, expected: 29.7, runtimeGiB: 6, downloadGiB: 2.7 }),
    ])
    expect(recommendations.map(({ displayName, intent }) => [displayName, intent])).toEqual([
      ["qwen35-q6", "balanced"],
      ["qwen27", "best_quality"],
      ["gemma26-100", "fastest"],
      ["qwen4", "lightweight"],
    ])
  })

  it("builds a useful DGX Spark-class portfolio around the strongest responsive model", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "laguna-100", checkpoint: "laguna", artifact: "laguna:q4", score: 70.2, fidelity: 40, expected: 26, context: 100_000, runtimeGiB: 90, downloadGiB: 73.4, capacityGiB: 121.7, architecture: "moe" }),
      candidate({ id: "laguna-200", checkpoint: "laguna", artifact: "laguna:q4", score: 70.2, fidelity: 40, expected: 14, context: 200_000, runtimeGiB: 104, downloadGiB: 73.4, capacityGiB: 121.7, architecture: "moe" }),
      candidate({ id: "qwen122", score: 47.6, fidelity: 40, expected: 16.87, context: 100_000, runtimeGiB: 84, downloadGiB: 71, capacityGiB: 121.7, architecture: "moe" }),
      candidate({ id: "gemma26", score: 39, fidelity: 58, expected: 42.7, context: 100_000, runtimeGiB: 28, downloadGiB: 13.3, capacityGiB: 121.7, architecture: "moe" }),
      candidate({ id: "qwen4", score: 25.8, fidelity: 40, expected: 21, context: 200_000, runtimeGiB: 6, downloadGiB: 2.7, capacityGiB: 121.7 }),
    ])

    expect(recommendations.map(({ displayName, intent }) => [displayName, intent])).toEqual([
      ["laguna-200", "balanced"],
      ["gemma26", "fastest"],
      ["qwen4", "lightweight"],
    ])
  })

  it("lets responsiveness outweigh a modest capability lead inside the Balanced guard", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "benchmark-leader", score: 60, expected: 16, runtimeGiB: 36 }),
      candidate({ id: "responsive", score: 48, expected: 45, runtimeGiB: 28 }),
    ])
    expect(byIntent(recommendations, "balanced")?.displayName).toBe("responsive")
  })

  it("still produces a useful portfolio when only small-machine candidates fit", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "small-quality", score: 40, fidelity: 80, expected: 32, runtimeGiB: 8, downloadGiB: 5 }),
      candidate({ id: "small-fast", score: 25.8, fidelity: 40, expected: 40, runtimeGiB: 6, downloadGiB: 3 }),
    ])
    expect(byIntent(recommendations, "balanced")?.displayName).toBe("small-quality")
    expect(byIntent(recommendations, "fastest")?.displayName).toBe("small-fast")
  })

  it("keeps multiple quantizations of one checkpoint when they serve different intents", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "q6", checkpoint: "same", artifact: "same:q6", score: 50, fidelity: 60, expected: 35, runtimeGiB: 25 }),
      candidate({ id: "q8", checkpoint: "same", artifact: "same:q8", score: 50, fidelity: 80, expected: 33, runtimeGiB: 32 }),
    ])
    expect(recommendations.map(({ recommendableModelId, intent }) =>
      [recommendableModelId, intent])).toEqual([
      ["same:q6", "balanced"],
      ["same:q8", "best_quality"],
    ])
  })

  it("lets Fastest select a materially quicker 100K profile", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, expected: 30, context: 200_000, runtimeGiB: 30 }),
      candidate({ id: "quick-100", checkpoint: "quick", artifact: "quick:q6", score: 45, expected: 50, context: 100_000, runtimeGiB: 28 }),
      candidate({ id: "quick-200", checkpoint: "quick", artifact: "quick:q6", score: 45, expected: 32, context: 200_000, runtimeGiB: 30 }),
    ])
    expect(byIntent(recommendations, "fastest")?.configuration.profile.contextLength).toBe(100_000)
  })

  it("uses confidence-aware conservative speed for Fastest", () => {
    const low = candidate({ id: "low-confidence", score: 45, expected: 100, lower: 16, confidence: "low" })
    const high = candidate({ id: "high-confidence", score: 45, expected: 50, lower: 40, confidence: "high" })
    expect(conservativeGenerationSpeed(low)).toBe(16)
    expect(conservativeGenerationSpeed(high)).toBe(50)
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, expected: 30 }),
      low,
      high,
    ])
    expect(byIntent(recommendations, "fastest")?.displayName).toBe("high-confidence")
  })

  it("does not apply a hidden discount to an explicit estimate", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "measured", score: 30, provenance: "measured_terminal_bench_2.1", expected: 25, runtimeGiB: 30 }),
      candidate({ id: "estimated", score: 30, provenance: "estimated_terminal_bench_2.1", expected: 40, runtimeGiB: 20 }),
    ])
    expect(byIntent(recommendations, "balanced")?.displayName).toBe("estimated")
  })

  it("keeps unmeasured models as fallback without letting them outrank scored models", () => {
    const scored = candidate({ id: "scored", score: 20, expected: 20 })
    const unmeasured = candidate({ id: "unmeasured", expected: 100, runtimeGiB: 2 })
    expect(byIntent(selectRecommendationPortfolio([scored, unmeasured]), "balanced")?.displayName)
      .toBe("scored")
    expect(byIntent(selectRecommendationPortfolio([unmeasured]), "balanced")?.displayName)
      .toBe("unmeasured")
  })

  it("does not emit duplicate filler intents", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "only", score: 50, expected: 30 }),
    ])
    expect(recommendations).toHaveLength(1)
    expect(recommendations[0]?.intent).toBe("balanced")
  })

  it("treats dense and MoE candidates only through their estimated vectors", () => {
    const dense = candidate({ id: "dense", score: 40, expected: 30, architecture: "dense" })
    const moe = candidate({ id: "moe", score: 40, expected: 30, architecture: "moe" })
    expect(selectRecommendationPortfolio([dense])[0]?.displayName).toBe("dense")
    expect(selectRecommendationPortfolio([moe])[0]?.displayName).toBe("moe")
  })

  it("keeps Fastest explanations consistent with the selected speed evidence", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 60, expected: 30 }),
      candidate({ id: "fast", score: 40, expected: 50 }),
    ])
    const fastest = byIntent(recommendations, "fastest")
    expect(fastest?.explanation).toContain("50.0 tokens/sec")
    expect(fastest?.explanation).toContain("67% faster than Balanced")
  })

  it("explains material trade-offs relative to Balanced", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, fidelity: 60, expected: 30, runtimeGiB: 30 }),
      candidate({ id: "quality", score: 56, fidelity: 80, expected: 24, runtimeGiB: 38 }),
      candidate({ id: "fast", score: 40, fidelity: 40, expected: 50, context: 100_000, runtimeGiB: 24 }),
      candidate({ id: "light", score: 32, fidelity: 40, expected: 35, runtimeGiB: 8, downloadGiB: 3 }),
    ])
    expect(byIntent(recommendations, "balanced")?.explanation).toContain("Best overall mix")
    expect(byIntent(recommendations, "best_quality")?.explanation).toContain("more memory than Balanced")
    expect(byIntent(recommendations, "best_quality")?.explanation).toContain("slower than Balanced")
    expect(byIntent(recommendations, "fastest")?.explanation)
      .toContain("half as much code and conversation history")
    expect(byIntent(recommendations, "fastest")?.explanation)
      .toContain("substantial compression")
    expect(byIntent(recommendations, "lightweight")?.explanation)
      .toContain("less capable on difficult coding tasks")
    expect(byIntent(recommendations, "lightweight")?.explanation)
      .toContain("faster than Balanced")
  })

  it("describes quantization quality absolutely, including quality-aware checkpoints", () => {
    const qatBase = candidate({
      id: "qat",
      score: 30,
      fidelity: 58,
      expected: 50,
      runtimeGiB: 20,
    })
    const qat = {
      ...qatBase,
      quantizationAware: true,
      model: { ...qatBase.model, quantizationAware: true },
    }
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, fidelity: 60, expected: 30, runtimeGiB: 30 }),
      qat,
      candidate({ id: "light", score: 30, fidelity: 40, expected: 25, runtimeGiB: 8 }),
    ])
    expect(byIntent(recommendations, "fastest")?.explanation)
      .toContain("very high output quality with minimal loss")
    expect(byIntent(recommendations, "fastest")?.explanation)
      .not.toContain("lower precision than Balanced")
    expect(byIntent(recommendations, "lightweight")?.explanation)
      .toContain("substantial compression with some possible quality loss")
  })
})
