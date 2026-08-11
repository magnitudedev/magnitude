import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  AssessmentEnvironmentIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelFileIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  ModelAssessmentIdSchema,
  RecommendableModelIdSchema,
} from "@magnitudedev/acn-protocol"
import {
  assembleRecommendationCatalogCandidates,
  balancedUtility,
  conservativeGenerationSpeed,
  selectRecommendationPortfolio,
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
  const checkpointId = input.checkpoint ?? input.id
  const artifactId = input.artifact ?? `${checkpointId}:q${input.fidelity ?? 60}`
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
      id: RecommendableModelIdSchema.make(artifactId),
      checkpointId,
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
            maximumContextLength: context,
          },
          },
        },
        profile,
      },
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
    checkpointId,
    capability: input.score === undefined
      ? undefined
      : {
          score: input.score,
          provenance: input.provenance ?? "measured_terminal_bench_2.1",
        },
    fidelityRank: fidelity,
    quantizationAware: false,
    estimatedLoadedBytes: runtimeBytes,
    stableCapacityBudgetBytes: capacityBytes,
    totalDownloadBytes: downloadBytes,
  }
}

const byIntent = (
  recommendations: readonly RecommendationSelection[],
  intent: RecommendationSelection["intent"],
): RecommendationSelection | undefined =>
  recommendations.find((recommendation) => recommendation.intent === intent)

describe("local model multicriteria recommendation policy", () => {
  it("excludes sub-floor measured speed from every intent", () => {
    const slow = candidate({
      id: "slow",
      score: 90,
      expected: 4.9,
    })
    expect(selectRecommendationPortfolio([slow])).toEqual([])
  })

  it("admits five expected tokens per second to recommendations", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "floor", score: 50, expected: 5 }),
    ])

    expect(byIntent(recommendations, "balanced")?.displayName).toBe("floor")
  })

  it("uses full-context speed for eligibility and 50K speed for comparisons", () => {
    const usable = candidate({
      id: "usable",
      score: 50,
      expected: 40,
      fullContextExpected: 5,
    })
    const tooSlowAtFullContext = candidate({
      id: "too-slow",
      score: 90,
      expected: 60,
      fullContextExpected: 4.9,
    })

    expect(conservativeGenerationSpeed(usable)).toBe(40)
    expect(selectRecommendationPortfolio([usable])).not.toEqual([])
    expect(selectRecommendationPortfolio([tooSlowAtFullContext])).toEqual([])
  })

  it("keeps compatible candidates below the recommendation speed floor in the catalog", () => {
    const slow = candidate({
      id: "slow",
      score: 80,
      expected: 4.9,
    })

    const recommendations = selectRecommendationPortfolio([slow])
    const catalog = assembleRecommendationCatalogCandidates([slow], recommendations)

    expect(recommendations).toEqual([])
    expect(catalog).toMatchObject([{ model: { displayName: "slow" } }])
  })

  it("applies the floor at the same one-decimal precision shown to users", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({
        id: "rounded-baseline",
        expected: 4.951,
      }),
    ])

    expect(byIntent(recommendations, "balanced")?.displayName).toBe("rounded-baseline")
  })

  it("awards speed utility above the five-token recommendation floor", () => {
    const five = candidate({ id: "five", expected: 5 })
    const eight = candidate({ id: "eight", expected: 8 })

    expect(balancedUtility(eight)).toBeGreaterThan(balancedUtility(five))
  })

  it("builds a useful 64 GiB-class portfolio and prefers capability inside Lightweight", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "qwen27", score: 60.7, fidelity: 50, expected: 10.9, context: 100_000, runtimeGiB: 26.68, downloadGiB: 18.9, capacityGiB: 57.6 }),
      candidate({ id: "qwen35-q6", checkpoint: "qwen35", artifact: "qwen35:q6", score: 44.9, fidelity: 60, expected: 36.4, runtimeGiB: 35.72, downloadGiB: 30.4, capacityGiB: 57.6, architecture: "moe" }),
      candidate({ id: "gemma26-100", checkpoint: "gemma26", score: 39, fidelity: 58, expected: 59.5, context: 100_000, runtimeGiB: 16.55, downloadGiB: 13.3, capacityGiB: 57.6, architecture: "moe" }),
      candidate({ id: "qwen4", score: 25.8, fidelity: 40, expected: 31.2, runtimeGiB: 11.25, downloadGiB: 2.8, capacityGiB: 57.6 }),
      candidate({ id: "gemma12", score: 21, fidelity: 58, expected: 29.8, runtimeGiB: 11.01, downloadGiB: 6.3, capacityGiB: 57.6 }),
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
      candidate({ id: "laguna-q4-100", checkpoint: "laguna", artifact: "laguna:q4", score: 70.2, fidelity: 40, expected: 12.1, context: 100_000, runtimeGiB: 73.41, downloadGiB: 68.4, capacityGiB: 109.5, architecture: "moe" }),
      candidate({ id: "laguna-q6", checkpoint: "laguna", artifact: "laguna:q6", score: 70.2, fidelity: 60, expected: 13.2, context: 100_000, runtimeGiB: 104.78, downloadGiB: 99.7, capacityGiB: 109.5, architecture: "moe" }),
      candidate({ id: "qwen35", score: 44.9, fidelity: 40, expected: 28.9, context: 100_000, runtimeGiB: 24.12, downloadGiB: 21.3, capacityGiB: 109.5, architecture: "moe" }),
      candidate({ id: "gemma26", score: 39, fidelity: 58, expected: 40.8, lower: 32.6, confidence: "moderate", context: 100_000, runtimeGiB: 16.55, downloadGiB: 13.3, capacityGiB: 109.5, architecture: "moe" }),
      candidate({ id: "qwen9", score: 29.2, fidelity: 40, expected: 18, context: 100_000, runtimeGiB: 12, downloadGiB: 5.7, capacityGiB: 109.5 }),
      candidate({ id: "qwen4", score: 25.8, fidelity: 40, expected: 22, context: 100_000, runtimeGiB: 8, downloadGiB: 2.8, capacityGiB: 109.5 }),
      candidate({ id: "gemma12", score: 21, fidelity: 58, expected: 20, context: 100_000, runtimeGiB: 9, downloadGiB: 6.3, capacityGiB: 109.5 }),
    ])

    expect(recommendations.map(({ displayName, intent }) => [displayName, intent])).toEqual([
      ["laguna-q4-100", "balanced"],
      ["laguna-q6", "best_quality"],
      ["gemma26", "fastest"],
      ["qwen9", "lightweight"],
    ])
  })

  it("does not let a new heavyweight capability ceiling downshift Lightweight", () => {
    const lightweightCandidates = [
      candidate({ id: "balanced", score: 50, expected: 40, runtimeGiB: 40, capacityGiB: 100 }),
      candidate({ id: "capable-light", score: 39, expected: 30, runtimeGiB: 18, capacityGiB: 100 }),
      candidate({ id: "tiny", score: 25.8, expected: 35, runtimeGiB: 8, capacityGiB: 100 }),
    ]
    const before = selectRecommendationPortfolio(lightweightCandidates)
    const after = selectRecommendationPortfolio([
      ...lightweightCandidates,
      candidate({ id: "heavyweight", score: 90, expected: 11, runtimeGiB: 85, capacityGiB: 100 }),
    ])

    expect(byIntent(before, "lightweight")?.displayName).toBe("capable-light")
    expect(byIntent(after, "lightweight")?.displayName).toBe("capable-light")
  })

  it("omits Lightweight when no unselected candidate is inside its memory tier", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, expected: 30, runtimeGiB: 50, capacityGiB: 100 }),
      candidate({ id: "smaller", score: 40, expected: 25, runtimeGiB: 25, capacityGiB: 100 }),
    ])

    expect(byIntent(recommendations, "lightweight")).toBeUndefined()
  })

  it("omits Lightweight when an in-tier candidate is not materially lighter than Balanced", () => {
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, expected: 40, runtimeGiB: 19, capacityGiB: 100 }),
      candidate({ id: "almost-as-heavy", score: 40, expected: 30, runtimeGiB: 15.5, capacityGiB: 100 }),
    ])

    expect(byIntent(recommendations, "lightweight")).toBeUndefined()
  })

  it("applies the Lightweight tier independently to each physical memory domain", () => {
    const splitBase = candidate({
      id: "device-heavy",
      score: 40,
      expected: 30,
      runtimeGiB: 18,
      capacityGiB: 100,
    })
    const split = {
      ...splitBase,
      assessment: {
        ...splitBase.assessment,
        memory: [
          {
            memoryDomainId: LocalInferenceMemoryDomainIdSchema.make("system"),
            capacityBytes: 80 * GIB,
            requiredBytes: 8 * GIB,
            compatibilityReserveBytes: 0,
            remainingBytes: 72 * GIB,
          },
          {
            memoryDomainId: LocalInferenceMemoryDomainIdSchema.make("device"),
            capacityBytes: 20 * GIB,
            requiredBytes: 10 * GIB,
            compatibilityReserveBytes: 0,
            remainingBytes: 10 * GIB,
          },
        ],
      },
    }
    const recommendations = selectRecommendationPortfolio([
      candidate({ id: "balanced", score: 50, expected: 40, runtimeGiB: 60, capacityGiB: 100 }),
      split,
    ])

    expect(byIntent(recommendations, "lightweight")).toBeUndefined()
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
      candidate({ id: "fast", score: 40, expected: 50.2 }),
    ])
    const fastest = byIntent(recommendations, "fastest")
    expect(fastest?.explanation).toContain("~50 tok/s at 50K context")
    expect(fastest?.explanation).not.toContain("50.2 tok/s")
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
      .toContain("Retains good quality with some possible loss")
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
      .toContain("Retains good quality with some possible loss")
  })
})
