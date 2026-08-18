import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  AssessmentEnvironmentIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelFileIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  ModelAssessmentIdSchema,
  ModelReleaseDateSchema,
  ModelVariantLabelSchema,
  CatalogModelIdSchema,
  CatalogVariantIdSchema,
} from "@magnitudedev/acn-protocol"
import {
  assembleRecommendationCatalogCandidates,
  balancedUtility,
  intentUtility,
  recommendationIntentWeights,
  selectRecommendationPortfolio,
  speedUtility,
  type RecommendationSelection,
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
  readonly context?: number
  readonly expected?: number
  readonly fullContextExpected?: number
  readonly lower?: number
  readonly upper?: number
  readonly confidence?: "high" | "moderate" | "low"
  readonly runtimeGiB?: number
  readonly downloadGiB?: number
  readonly capacityGiB?: number
  readonly architecture?: "dense" | "moe"
}): RecommendationCandidate => {
  const catalogModelId = input.checkpoint ?? input.id
  const artifactId = input.artifact ?? `${catalogModelId}:q${input.fidelity ?? 60}`
  const qualityTrack = artifactId.split(":").at(-1) ?? "q4"
  const context = input.context ?? 100_000
  const expected = input.expected ?? 30
  const fidelity = input.fidelity ?? 60
  const runtimeBytes = (input.runtimeGiB ?? 24) * GIB
  const downloadBytes = (input.downloadGiB ?? input.runtimeGiB ?? 20) * GIB
  const capacityBytes = (input.capacityGiB ?? 64) * GIB
  const packageId = ModelPackageIdSchema.make(`package_${input.id}`)
  const profile = { contextLength: context }
  const configurationId = ModelServingConfigurationIdSchema.make(`${input.id}:ctx${context}`)
  const comparisonContext = Math.min(50_000, context)
  const performanceContexts = [...new Set([
    ...[25_000, 50_000, 75_000].filter((sample) => sample <= context),
    context,
  ])].sort((left, right) => left - right)
  return {
    model: {
      modelId: CatalogModelIdSchema.make(catalogModelId),
      variantId: CatalogVariantIdSchema.make(`gguf:${qualityTrack}`),
      configuration: {
        id: configurationId,
        bundle: {
          _tag: "Standalone",
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
            tensorStorageBytes: Option.none(),
            sha256: "a".repeat(64),
          }],
          relationships: [],
          properties: {
            format: "gguf",
            quantization: `Q${fidelity}`,
            quantizationName: `${fidelity}-bit`,
            architecture: input.architecture ?? "dense",
            maximumContextLength: Option.some(context),
            intrinsicModelId: Option.some(catalogModelId),
            intrinsicQualityId: Option.some(`Q${fidelity}`),
          },
          },
        },
        profile,
      },
      displayName: input.id,
      variantLabel: ModelVariantLabelSchema.make(`Q${fidelity}`),
      description: "Test fixture",
      releaseDate: ModelReleaseDateSchema.make("2026-01-01"),
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
      parameterization: { architecture: "dense", totalParameters: 8_000_000_000 },
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
      assessmentId: ModelAssessmentIdSchema.make(`assessment_${input.id}_${context}`),
      environmentId: AssessmentEnvironmentIdSchema.make("environment_test"),
      memory: [{
        memoryDomainId: LocalInferenceMemoryDomainIdSchema.make("memory"),
        capacityBytes,
        requiredBytes: runtimeBytes,
        compatibilityReserveBytes: 0,
        remainingBytes: capacityBytes - runtimeBytes,
      }],
      performance: performanceContexts.map((contextTokens) => {
        const estimatedTokensPerSecond = contextTokens === context
          ? input.fullContextExpected ?? expected
          : expected
        return {
          contextTokens,
          lowerTokensPerSecond: contextTokens === comparisonContext
            ? input.lower ?? estimatedTokensPerSecond * 0.85
            : estimatedTokensPerSecond * 0.85,
          estimatedTokensPerSecond,
          upperTokensPerSecond: contextTokens === comparisonContext
            ? input.upper ?? estimatedTokensPerSecond * 1.15
            : estimatedTokensPerSecond * 1.15,
          confidence: input.confidence ?? "high",
        }
      }),
    },
    artifactId,
    catalogModelId,
    capabilityScore: input.score ?? 50,
    fidelityRank: fidelity,
    quantizationAware: false,
    estimatedLoadedBytes: runtimeBytes,
    stableCapacityBudgetBytes: capacityBytes,
  }
}

const byIntent = (
  recommendations: readonly RecommendationSelection[],
  intent: RecommendationSelection["intent"],
): RecommendationSelection | undefined =>
  recommendations.find((recommendation) => recommendation.intent === intent)

describe("local model multicriteria recommendation policy", () => {
  it("maps every intent to a weight vector over the same four factors", () => {
    expect(Object.keys(recommendationIntentWeights)).toEqual([
      "balanced",
      "smartest",
      "fastest",
      "lightweight",
    ])
    for (const weights of Object.values(recommendationIntentWeights)) {
      expect(Object.keys(weights)).toEqual(["capability", "speed", "fidelity", "memory"])
      expect(Object.values(weights).reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1)
    }
  })

  it("uses the 50K speed factor without a separate speed eligibility gate", () => {
    const slow = candidate({ id: "slow", score: 90, expected: 4.9 })
    const recommendations = selectRecommendationPortfolio([slow])

    expect(recommendations.map(({ intent }) => intent)).toEqual(["balanced"])
    expect(assembleRecommendationCatalogCandidates([slow], recommendations)).toHaveLength(1)
  })

  it("scores speed linearly through 40 tokens per second", () => {
    const maximumUtility = 1 + Math.log(100 / 40)

    expect(speedUtility(0)).toBe(0)
    expect(speedUtility(10)).toBeCloseTo(0.25 / maximumUtility)
    expect(speedUtility(40)).toBeCloseTo(1 / maximumUtility)
  })

  it("scores speed logarithmically from 40 to 100 tokens per second", () => {
    const maximumUtility = 1 + Math.log(100 / 40)
    const expectedAt60 = (1 + Math.log(60 / 40)) / maximumUtility

    expect(speedUtility(60)).toBeCloseTo(expectedAt60)
    expect(speedUtility(100)).toBe(1)
    expect(speedUtility(200)).toBe(1)
  })

  it("lets speed proportionally outweigh capability without a capability gate", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "qwen38-q8", score: 73, fidelity: 80, expected: 10 }),
      candidate({ id: "qwen36-35b-a3b-q6", score: 44.9, fidelity: 60, expected: 40 }),
    ])

    expect(byIntent(recommendations, "balanced")?.displayName)
      .toBe("qwen36-35b-a3b-q6")
  })

  it("prefers Q6 fidelity over a merely ten-percent-faster Q5", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "q6", score: 50, fidelity: 60, expected: 30 }),
      candidate({ id: "q5", score: 50, fidelity: 50, expected: 33 }),
    ])

    expect(byIntent(recommendations, "balanced")?.displayName).toBe("q6")
  })

  it("allows a substantially faster Q5 to outweigh Q6 fidelity", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "q6", score: 50, fidelity: 60, expected: 30 }),
      candidate({ id: "q5", score: 50, fidelity: 50, expected: 40 }),
    ])

    expect(byIntent(recommendations, "balanced")?.displayName).toBe("q5")
  })

  it("does not include download size in Balanced utility", () => {
    const smallDownload = candidate({
      id: "small-download",
      score: 50,
      fidelity: 60,
      expected: 30,
      downloadGiB: 1,
    })
    const largeDownload = candidate({
      id: "large-download",
      score: 50,
      fidelity: 60,
      expected: 30,
      downloadGiB: 100,
    })

    expect(balancedUtility(smallDownload)).toBe(balancedUtility(largeDownload))
  })

  it("lets Smartest trade all factors using its intelligence-and-fidelity-heavy weights", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, fidelity: 60, expected: 40, runtimeGiB: 20 }),
      candidate({ id: "quality-gain", score: 52, fidelity: 75, expected: 30, runtimeGiB: 24 }),
    ])

    expect(byIntent(recommendations, "balanced")?.displayName).toBe("balanced")
    expect(byIntent(recommendations, "smartest")?.displayName).toBe("quality-gain")
  })

  it("applies all four objectives to a representative 64 GiB portfolio", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "qwen27", score: 60.7, fidelity: 50, expected: 10.9, context: 100_000, runtimeGiB: 26.68, downloadGiB: 18.9, capacityGiB: 57.6 }),
      candidate({ id: "qwen35-q6", checkpoint: "qwen35", artifact: "qwen35:q6", score: 44.9, fidelity: 60, expected: 36.4, runtimeGiB: 35.72, downloadGiB: 30.4, capacityGiB: 57.6, architecture: "moe" }),
      candidate({ id: "gemma26-100", checkpoint: "gemma26", score: 39, fidelity: 58, expected: 59.5, context: 100_000, runtimeGiB: 16.55, downloadGiB: 13.3, capacityGiB: 57.6, architecture: "moe" }),
      candidate({ id: "qwen4", score: 25.8, fidelity: 40, expected: 31.2, runtimeGiB: 11.25, downloadGiB: 2.8, capacityGiB: 57.6 }),
      candidate({ id: "gemma12", score: 21, fidelity: 58, expected: 29.8, runtimeGiB: 11.01, downloadGiB: 6.3, capacityGiB: 57.6 }),
    ])
    expect(recommendations.map(({ displayName, intent }) => [displayName, intent])).toEqual([
      ["gemma26-100", "balanced"],
      ["qwen27", "smartest"],
      ["qwen35-q6", "fastest"],
      ["gemma12", "lightweight"],
    ])
  })

  it("builds a useful DGX Spark-class portfolio around the strongest responsive model", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "laguna-q4-100", checkpoint: "laguna", artifact: "laguna:q4", score: 70.2, fidelity: 40, expected: 12.1, context: 100_000, runtimeGiB: 73.41, downloadGiB: 68.4, capacityGiB: 109.5, architecture: "moe" }),
      candidate({ id: "laguna-q6", checkpoint: "laguna", artifact: "laguna:q6", score: 70.2, fidelity: 60, expected: 13.2, context: 100_000, runtimeGiB: 104.78, downloadGiB: 99.7, capacityGiB: 109.5, architecture: "moe" }),
      candidate({ id: "qwen35", score: 44.9, fidelity: 40, expected: 28.9, context: 100_000, runtimeGiB: 24.12, downloadGiB: 21.3, capacityGiB: 109.5, architecture: "moe" }),
      candidate({ id: "gemma26", score: 39, fidelity: 58, expected: 40.8, lower: 32.6, confidence: "moderate", context: 100_000, runtimeGiB: 16.55, downloadGiB: 13.3, capacityGiB: 109.5, architecture: "moe" }),
      candidate({ id: "qwen9", score: 29.2, fidelity: 40, expected: 18, context: 100_000, runtimeGiB: 12, downloadGiB: 5.7, capacityGiB: 109.5 }),
      candidate({ id: "qwen4", score: 25.8, fidelity: 40, expected: 22, context: 100_000, runtimeGiB: 8, downloadGiB: 2.8, capacityGiB: 109.5 }),
      candidate({ id: "gemma12", score: 21, fidelity: 58, expected: 20, context: 100_000, runtimeGiB: 9, downloadGiB: 6.3, capacityGiB: 109.5 }),
    ])

    expect(recommendations.map(({ displayName, intent }) => [displayName, intent])).toEqual([
      ["gemma26", "balanced"],
      ["laguna-q6", "smartest"],
      ["qwen35", "fastest"],
      ["gemma12", "lightweight"],
    ])
  })

  it("lets Lightweight rank naturally through its memory-heavy weights", () => {
    expect(intentUtility(candidate({ id: "tiny-utility", score: 25.8, expected: 30, runtimeGiB: 8, capacityGiB: 100 }), "lightweight"))
      .toBeGreaterThan(intentUtility(candidate({ id: "heavy-utility", score: 70, expected: 40, runtimeGiB: 70, capacityGiB: 100 }), "lightweight"))
  })

  it("lets responsiveness proportionally outweigh a modest capability lead", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "benchmark-leader", score: 60, expected: 16, runtimeGiB: 36 }),
      candidate({ id: "responsive", score: 48, expected: 45, runtimeGiB: 28 }),
    ])
    expect(byIntent(recommendations, "balanced")?.displayName).toBe("responsive")
  })

  it("emits only as many ordered intents as there are distinct candidates", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "small-quality", score: 40, fidelity: 80, expected: 32, runtimeGiB: 8, downloadGiB: 5 }),
      candidate({ id: "small-fast", score: 25.8, fidelity: 40, expected: 40, runtimeGiB: 6, downloadGiB: 3 }),
    ])
    expect(byIntent(recommendations, "balanced")?.displayName).toBe("small-quality")
    expect(byIntent(recommendations, "smartest")?.displayName).toBe("small-fast")
    expect(recommendations).toHaveLength(2)
  })

  it("keeps multiple quantizations of one checkpoint when they serve different intents", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "q6", checkpoint: "same", artifact: "same:q6", score: 50, fidelity: 60, expected: 40, runtimeGiB: 25 }),
      candidate({ id: "q8", checkpoint: "same", artifact: "same:q8", score: 50, fidelity: 80, expected: 33, runtimeGiB: 32 }),
    ])
    expect(recommendations.map(({ recommendableModelId, intent }) =>
      [recommendableModelId, intent])).toEqual([
      ["same:gguf:q6", "balanced"],
      ["same:gguf:q8", "smartest"],
    ])
  })

  it("uses the same explicit speed estimate for Fastest regardless of provenance confidence", () => {
    const low = candidate({ id: "low-confidence", score: 45, expected: 100, lower: 16, confidence: "low" })
    const high = candidate({ id: "high-confidence", score: 45, expected: 50, lower: 40, confidence: "high" })
    expect(intentUtility(low, "fastest")).toBeGreaterThan(intentUtility(high, "fastest"))
  })

  it("does not apply a hidden discount to an explicit estimate", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "measured", score: 30, provenance: "measured_terminal_bench_2.1", expected: 25, runtimeGiB: 30 }),
      candidate({ id: "estimated", score: 30, provenance: "estimated_terminal_bench_2.1", expected: 40, runtimeGiB: 20 }),
    ])
    expect(byIntent(recommendations, "balanced")?.displayName).toBe("estimated")
  })

  it("never assigns one configuration to multiple intents", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "only", score: 50, expected: 30 }),
    ])
    expect(recommendations.map(({ intent }) => intent)).toEqual(["balanced"])
    expect(new Set(recommendations.map(({ configurationId }) => configurationId)).size).toBe(1)
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
      candidate({ id: "smartest", score: 90, fidelity: 80, expected: 10 }),
      candidate({ id: "fast", score: 30, expected: 50.2 }),
    ])
    const fastest = byIntent(recommendations, "fastest")
    expect(fastest?.explanation).toContain("~50 tok/s at 50K context")
    expect(fastest?.explanation).not.toContain("50.2 tok/s")
    expect(fastest?.explanation).toContain("67% faster than Balanced")
  })

  it("explains material trade-offs relative to Balanced", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, fidelity: 60, expected: 30, runtimeGiB: 30 }),
      candidate({ id: "quality", score: 56, fidelity: 80, expected: 15, runtimeGiB: 38 }),
      candidate({ id: "fast", score: 40, fidelity: 40, expected: 50, context: 100_000, runtimeGiB: 24 }),
      candidate({ id: "light", score: 32, fidelity: 40, expected: 35, runtimeGiB: 8, downloadGiB: 3 }),
    ])
    expect(byIntent(recommendations, "balanced")?.explanation).toContain("Best overall mix")
    expect(byIntent(recommendations, "smartest")?.explanation).toContain("more memory than Balanced")
    expect(byIntent(recommendations, "smartest")?.explanation).toContain("slower than Balanced")
    expect(byIntent(recommendations, "fastest")?.explanation)
      .toContain("Retains good quality with some possible loss")
    expect(byIntent(recommendations, "lightweight")?.explanation)
      .toContain("less capable on difficult coding tasks")
    expect(byIntent(recommendations, "lightweight")?.explanation)
      .toContain("faster than Balanced")
  })

  it("describes quantization quality absolutely, including quality-aware checkpoints", () => {
    const qatBase = candidate({
      id: "qat",
      score: 25,
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
      candidate({ id: "smart", score: 90, fidelity: 80, expected: 10, runtimeGiB: 40 }),
      qat,
      candidate({ id: "light", score: 25, fidelity: 40, expected: 25, runtimeGiB: 8 }),
    ])
    expect(byIntent(recommendations, "fastest")?.explanation)
      .toContain("very high output quality with minimal loss")
    expect(byIntent(recommendations, "fastest")?.explanation)
      .not.toContain("lower precision than Balanced")
    expect(byIntent(recommendations, "lightweight")?.explanation)
      .toContain("Retains good quality with some possible loss")
  })
})
