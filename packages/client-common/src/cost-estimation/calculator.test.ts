import { describe, expect, it } from "vitest"
import { Either, Option } from "effect"
import {
  CurrencyCode,
  PricingEntry,
  PricingModelId,
  PricingOverride,
  PricingProviderId,
} from "./types"
import {
  aggregateMultiModelCost,
  calculateModelCost,
  estimateHistoricalDailyCosts,
  roundCost,
} from "./calculator"
import { PricingRegistry } from "./registry"
import { extractBaseModelId, resolveModelPricing } from "./equivalence"

const USD = CurrencyCode.make("USD")
const EUR = CurrencyCode.make("EUR")

// Deterministic test fixtures
const sonnetPricing: PricingEntry = {
  providerId: PricingProviderId.make("anthropic"),
  modelId: PricingModelId.make("claude-3-5-sonnet-20241022"),
  displayName: "Claude 3.5 Sonnet",
  inputPricePerMillion: 3.0,
  cachedInputPricePerMillion: Option.some(0.3),
  outputPricePerMillion: 15.0,
  currency: USD,
  pricingSourceUrl: "https://www.anthropic.com/pricing",
  lastVerified: "2025-01-15",
  effectiveDate: Option.none(),
  isEstimated: false,
}

// Pricing fixture without cached input pricing (fallback to standard input rate)
const noCacheDiscountPricing: PricingEntry = {
  providerId: PricingProviderId.make("together"),
  modelId: PricingModelId.make("llama-3.3-70b-instruct"),
  displayName: "Llama 3.3 70B Instruct (Hosted)",
  inputPricePerMillion: 0.88,
  cachedInputPricePerMillion: Option.none(),
  outputPricePerMillion: 0.88,
  currency: USD,
  pricingSourceUrl: "https://www.together.ai/pricing",
  lastVerified: "2025-01-15",
  effectiveDate: Option.none(),
  isEstimated: false,
}

describe("API-Equivalent Cost Calculation", () => {
  // 1. Standard input and output cost
  it("calculates standard input and output costs correctly", () => {
    // 100,000 input tokens at $3.00/1M = $0.30
    // 50,000 output tokens at $15.00/1M = $0.75
    // Total = $1.05
    const result = calculateModelCost(
      { inputTokens: 100_000, cachedInputTokens: 0, outputTokens: 50_000 },
      sonnetPricing,
    )
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      expect(result.right.uncachedInputCost).toBe(0.3)
      expect(result.right.cachedInputCost).toBe(0)
      expect(result.right.outputCost).toBe(0.75)
      expect(result.right.totalCost).toBe(1.05)
      expect(result.right.uncachedInputTokens).toBe(100_000)
      expect(result.right.cachedInputTokens).toBe(0)
      expect(result.right.outputTokens).toBe(50_000)
      expect(result.right.totalTokens).toBe(150_000)
      expect(result.right.cacheHitPercentage).toBe(0)
      expect(result.right.currency).toBe(USD)
    }
  })

  // 2. Cached input pricing
  it("calculates cached input pricing without double counting", () => {
    // Total input: 100,000 tokens (40,000 cached @ $0.30/1M, 60,000 uncached @ $3.00/1M)
    // 60,000 uncached * 3.0 / 1M = $0.18
    // 40,000 cached * 0.3 / 1M = $0.012
    // 20,000 output * 15.0 / 1M = $0.30
    // Total = $0.18 + $0.012 + $0.30 = $0.492
    const result = calculateModelCost(
      { inputTokens: 100_000, cachedInputTokens: 40_000, outputTokens: 20_000 },
      sonnetPricing,
    )
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      expect(result.right.uncachedInputTokens).toBe(60_000)
      expect(result.right.cachedInputTokens).toBe(40_000)
      expect(result.right.outputTokens).toBe(20_000)
      expect(result.right.totalTokens).toBe(120_000)
      expect(result.right.uncachedInputCost).toBe(0.18)
      expect(result.right.cachedInputCost).toBe(0.012)
      expect(result.right.outputCost).toBe(0.3)
      expect(result.right.totalCost).toBe(0.492)
      expect(result.right.cacheHitPercentage).toBe(40)
      expect(result.right.cachedInputPricingAvailable).toBe(true)
      expect(result.right.effectiveCachedInputPricePerMillion).toBe(0.3)
    }
  })

  // 3. Zero cached tokens
  it("handles zero cached tokens properly", () => {
    const result = calculateModelCost(
      { inputTokens: 50_000, cachedInputTokens: 0, outputTokens: 10_000 },
      sonnetPricing,
    )
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      expect(result.right.cachedInputTokens).toBe(0)
      expect(result.right.cachedInputCost).toBe(0)
      expect(result.right.uncachedInputTokens).toBe(50_000)
      expect(result.right.uncachedInputCost).toBe(0.15)
      expect(result.right.cacheHitPercentage).toBe(0)
    }
  })

  // 4. All tokens cached
  it("handles 100% cached tokens properly", () => {
    // 100,000 total input, 100,000 cached
    // Uncached = 0 tokens, $0.00
    // Cached = 100,000 * 0.3 / 1M = $0.03
    // Output = 10,000 * 15.0 / 1M = $0.15
    // Total = $0.18
    const result = calculateModelCost(
      { inputTokens: 100_000, cachedInputTokens: 100_000, outputTokens: 10_000 },
      sonnetPricing,
    )
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      expect(result.right.uncachedInputTokens).toBe(0)
      expect(result.right.uncachedInputCost).toBe(0)
      expect(result.right.cachedInputTokens).toBe(100_000)
      expect(result.right.cachedInputCost).toBe(0.03)
      expect(result.right.totalCost).toBe(0.18)
      expect(result.right.cacheHitPercentage).toBe(100)
    }
  })

  // 5. Zero-token usage
  it("handles zero-token requests safely", () => {
    const result = calculateModelCost(
      { inputTokens: 0, cachedInputTokens: 0, outputTokens: 0 },
      sonnetPricing,
    )
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      expect(result.right.uncachedInputCost).toBe(0)
      expect(result.right.cachedInputCost).toBe(0)
      expect(result.right.outputCost).toBe(0)
      expect(result.right.totalCost).toBe(0)
      expect(result.right.totalTokens).toBe(0)
      expect(result.right.cacheHitPercentage).toBe(0)
    }
  })

  // 6. Missing cached-input pricing (fallback policy)
  it("applies transparent fallback policy when cached input pricing is unavailable", () => {
    // Model has inputPrice = $0.88, no cached input discount (cachedInputPricePerMillion = None).
    // 100,000 total input, 40,000 cached.
    // Fallback charges cached tokens at the standard input price ($0.88/1M).
    // Uncached: 60,000 * 0.88 / 1M = 0.0528
    // Cached: 40,000 * 0.88 / 1M = 0.0352
    // Sum input: 0.0528 + 0.0352 = 0.088 (exact same as 100,000 * 0.88 / 1M)
    // Output: 10,000 * 0.88 / 1M = 0.0088
    // Total = 0.088 + 0.0088 = 0.0968
    const result = calculateModelCost(
      { inputTokens: 100_000, cachedInputTokens: 40_000, outputTokens: 10_000 },
      noCacheDiscountPricing,
    )
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      expect(result.right.cachedInputPricingAvailable).toBe(false)
      expect(result.right.effectiveCachedInputPricePerMillion).toBe(0.88)
      expect(result.right.uncachedInputCost).toBe(0.0528)
      expect(result.right.cachedInputCost).toBe(0.0352)
      expect(result.right.totalCost).toBe(0.0968)
    }
  })

  // 7. Unknown model pricing
  it("handles unknown model pricing gracefully without fabricating costs", () => {
    const registry = new PricingRegistry([])
    const resolution = resolveModelPricing("completely-unknown-custom-model-xyz", { registry })
    expect(resolution._tag).toBe("Unmapped")
    if (resolution._tag === "Unmapped") {
      expect(resolution.modelId).toBe("completely-unknown-custom-model-xyz")
      expect(resolution.reason).toContain("No hosted equivalent configured")
    }
  })

  // 8. Invalid negative token counts
  it("rejects invalid negative token counts with an explicit error", () => {
    const result = calculateModelCost(
      { inputTokens: -100, cachedInputTokens: 0, outputTokens: 50 },
      sonnetPricing,
    )
    expect(Either.isLeft(result)).toBe(true)
    if (Either.isLeft(result)) {
      expect(result.left._tag).toBe("InvalidUsageDataError")
      expect(result.left.message).toContain("Token counts cannot be negative")
    }
  })

  // 9. Cached input exceeding total input
  it("safely clamps cached tokens when reported cached tokens exceed total input", () => {
    // 50,000 total input, but 80,000 cached reported
    // Clamps cached to 50,000, uncached becomes 0.
    const result = calculateModelCost(
      { inputTokens: 50_000, cachedInputTokens: 80_000, outputTokens: 10_000 },
      sonnetPricing,
    )
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      expect(result.right.cachedInputTokens).toBe(50_000)
      expect(result.right.uncachedInputTokens).toBe(0)
      expect(result.right.uncachedInputCost).toBe(0)
      expect(result.right.cachedInputCost).toBe(0.015) // 50,000 * 0.3 / 1M
      expect(result.right.cacheHitPercentage).toBe(100)
    }
  })

  // 10. Aggregation across multiple models
  it("aggregates usage and costs across multiple models with full breakdown", () => {
    const registry = new PricingRegistry([sonnetPricing, noCacheDiscountPricing])
    const items = [
      {
        modelId: "claude-3-5-sonnet-20241022",
        inputTokens: 100_000,
        cachedInputTokens: 40_000,
        outputTokens: 20_000,
        requests: 5,
      },
      {
        modelId: "llama-3.3-70b-instruct",
        inputTokens: 50_000,
        cachedInputTokens: 0,
        outputTokens: 10_000,
        requests: 2,
      },
    ]

    const result = aggregateMultiModelCost(items, { registry })
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      // Model 1:
      // uncachedCost = 60,000 * 3.0 / 1M = 0.18
      // cachedCost = 40,000 * 0.3 / 1M = 0.012
      // outputCost = 20,000 * 15.0 / 1M = 0.30
      // subtotal 1 = 0.492
      // Model 2:
      // uncachedCost = 50,000 * 0.88 / 1M = 0.044
      // cachedCost = 0
      // outputCost = 10,000 * 0.88 / 1M = 0.0088
      // subtotal 2 = 0.0528
      // Aggregate:
      // total = 0.492 + 0.0528 = 0.5448
      expect(result.right.totalCost).toBe(0.5448)
      expect(result.right.requests).toBe(7)
      expect(result.right.uncachedInputTokens).toBe(110_000)
      expect(result.right.cachedInputTokens).toBe(40_000)
      expect(result.right.outputTokens).toBe(30_000)
      expect(result.right.totalTokens).toBe(180_000)
      expect(result.right.isComplete).toBe(true)
      expect(result.right.modelsCalculated.length).toBe(2)
      expect(result.right.modelsMissingPricing.length).toBe(0)
    }
  })

  it("handles partial multi-model aggregation when some models have missing pricing", () => {
    const registry = new PricingRegistry([sonnetPricing])
    const items = [
      {
        modelId: "claude-3-5-sonnet-20241022",
        inputTokens: 100_000,
        cachedInputTokens: 0,
        outputTokens: 10_000,
        requests: 1,
      },
      {
        modelId: "unpriced-custom-model",
        inputTokens: 50_000,
        cachedInputTokens: 0,
        outputTokens: 5_000,
        requests: 1,
      },
    ]

    const result = aggregateMultiModelCost(items, { registry })
    expect(Either.isRight(result)).toBe(true)
    if (Either.isRight(result)) {
      expect(result.right.isComplete).toBe(false)
      expect(result.right.modelsCalculated.length).toBe(1)
      expect(result.right.modelsMissingPricing.length).toBe(1)
      expect(result.right.modelsMissingPricing[0]?.modelId).toBe("unpriced-custom-model")
    }
  })

  // 11. Historical usage aggregation
  it("projects historical daily API-equivalent costs proportionately over daily activity", () => {
    const dailyActivity = [
      { date: "2026-03-01", totalTokens: 100_000 },
      { date: "2026-03-02", totalTokens: 200_000 },
      { date: "2026-03-03", totalTokens: 0 },
      { date: "2026-03-04", totalTokens: 100_000 },
    ]
    const totalTokens = 400_000
    const totalCost = 4.0 // $4.00 total across the 400,000 tokens ($10 per 1M effective)

    const dailyEstimates = estimateHistoricalDailyCosts(dailyActivity, totalTokens, totalCost, USD)
    expect(dailyEstimates.length).toBe(4)
    expect(dailyEstimates[0]?.estimatedCost).toBe(1.0)
    expect(dailyEstimates[1]?.estimatedCost).toBe(2.0)
    expect(dailyEstimates[2]?.estimatedCost).toBe(0.0)
    expect(dailyEstimates[3]?.estimatedCost).toBe(1.0)
    expect(dailyEstimates[0]?.currency).toBe(USD)
  })

  // 12. Currency handling
  it("handles currency propagation and rejects mixed currencies during aggregation", () => {
    const eurPricing: PricingEntry = {
      ...sonnetPricing,
      modelId: PricingModelId.make("claude-in-eur"),
      currency: EUR,
    }
    const registry = new PricingRegistry([sonnetPricing, eurPricing])

    const items = [
      { modelId: "claude-3-5-sonnet-20241022", inputTokens: 10_000, cachedInputTokens: 0, outputTokens: 1_000 },
      { modelId: "claude-in-eur", inputTokens: 10_000, cachedInputTokens: 0, outputTokens: 1_000 },
    ]

    const result = aggregateMultiModelCost(items, { registry })
    expect(Either.isLeft(result)).toBe(true)
    if (Either.isLeft(result)) {
      expect(result.left._tag).toBe("CurrencyMismatchError")
      expect(result.left.message).toContain("Cannot aggregate models with different currencies")
    }
  })

  // 13. Custom pricing overrides
  it("applies custom pricing overrides with highest precedence", () => {
    const override: PricingOverride = {
      displayName: Option.some("My Custom Discounted Claude"),
      inputPricePerMillion: 1.5, // 50% discount override
      cachedInputPricePerMillion: Option.some(0.15),
      outputPricePerMillion: 7.5,
      currency: Option.some(USD),
      pricingSourceUrl: Option.some("https://custom.override/pricing"),
      isEstimated: Option.some(true),
    }

    const resolution = resolveModelPricing("claude-3-5-sonnet-20241022", {
      overrides: {
        "claude-3-5-sonnet-20241022": override,
      },
    })

    expect(resolution._tag).toBe("Resolved")
    if (resolution._tag === "Resolved") {
      expect(resolution.source).toBe("user_override")
      expect(resolution.entry.displayName).toBe("My Custom Discounted Claude")
      expect(resolution.entry.inputPricePerMillion).toBe(1.5)
      expect(resolution.entry.outputPricePerMillion).toBe(7.5)
      expect(resolution.entry.isEstimated).toBe(true)

      const costResult = calculateModelCost(
        { inputTokens: 1_000_000, cachedInputTokens: 0, outputTokens: 1_000_000 },
        resolution.entry,
      )
      expect(Either.isRight(costResult)).toBe(true)
      if (Either.isRight(costResult)) {
        // 1M * 1.5/1M + 1M * 7.5/1M = 1.5 + 7.5 = 9.0
        expect(costResult.right.totalCost).toBe(9.0)
      }
    }
  })

  // Precision and floating point checks
  it("avoids floating point drift in rounding calculations", () => {
    // 1 token at $3.00/1M = 0.000003 dollars
    expect(roundCost(0.1 + 0.2)).toBe(0.3)
    const cost = calculateModelCost(
      { inputTokens: 1, cachedInputTokens: 0, outputTokens: 0 },
      sonnetPricing,
    )
    expect(Either.isRight(cost)).toBe(true)
    if (Either.isRight(cost)) {
      expect(cost.right.uncachedInputCost).toBe(0.000003)
      expect(cost.right.totalCost).toBe(0.000003)
    }
  })

  // Model equivalence resolution
  it("normalizes model IDs and resolves explicit mapping", () => {
    expect(extractBaseModelId("llama-3.3-70b-instruct:gguf:q4_k_m")).toBe("llama-3.3-70b-instruct")
    expect(extractBaseModelId("unsloth/DeepSeek-V4-Flash-0731-GGUF")).toBe("deepseek-v4")
    expect(extractBaseModelId("magnitudedev/Qwen3.5-122B-A10B-MTP-GGUF")).toBe("qwen3.5-122b")

    const res = resolveModelPricing("llama-3.3-70b-instruct:gguf:q4_k_m")
    expect(res._tag).toBe("Resolved")
    if (res._tag === "Resolved") {
      expect(res.source).toBe("explicit_mapping")
      expect(res.entry.modelId).toBe("llama-3.3-70b-instruct")
    }
  })

  it("resolves user-selected comparison model", () => {
    const res = resolveModelPricing("any-custom-fine-tune:gguf:q4", {
      selectedComparisonModelId: "gpt-4o",
    })
    expect(res._tag).toBe("Resolved")
    if (res._tag === "Resolved") {
      expect(res.source).toBe("user_selected")
      expect(res.entry.modelId).toBe("gpt-4o")
      expect(res.entry.displayName).toBe("GPT-4o")
    }
  })
})
