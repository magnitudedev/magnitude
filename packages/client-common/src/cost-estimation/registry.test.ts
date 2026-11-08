import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import {
  BUILT_IN_PRICING,
  defaultPricingRegistry,
  PricingRegistry,
  applyPricingOverride,
} from "./registry"
import {
  CurrencyCode,
  PricingEntry,
  PricingModelId,
  PricingOverride,
  PricingProviderId,
} from "./types"

describe("PricingRegistry", () => {
  it("validates all built-in pricing entries against PricingEntry Schema", () => {
    expect(BUILT_IN_PRICING.length).toBeGreaterThan(5)
    for (const entry of BUILT_IN_PRICING) {
      const encoded = Schema.encodeSync(PricingEntry)(entry)
      const decoded = Schema.decodeSync(PricingEntry)(encoded)
      expect(decoded.modelId).toBe(entry.modelId)
      expect(decoded.inputPricePerMillion).toBeGreaterThanOrEqual(0)
      expect(decoded.outputPricePerMillion).toBeGreaterThanOrEqual(0)
      expect(decoded.currency).toBe("USD")
      expect(decoded.pricingSourceUrl).toMatch(/^https?:\/\//)
      expect(decoded.lastVerified).toMatch(/^\d{4}-\d{2}-\d{2}$/)
      expect(typeof decoded.isEstimated).toBe("boolean")
    }
  })

  it("lists all registered models", () => {
    const list = defaultPricingRegistry.list()
    expect(list.length).toBe(BUILT_IN_PRICING.length)
    expect(defaultPricingRegistry.has("claude-3-5-sonnet-20241022")).toBe(true)
    expect(defaultPricingRegistry.has("gpt-4o")).toBe(true)
    expect(defaultPricingRegistry.has("deepseek-chat")).toBe(true)
  })

  it("supports registering custom entries dynamically", () => {
    const registry = new PricingRegistry([])
    const customEntry: PricingEntry = {
      providerId: PricingProviderId.make("custom-provider"),
      modelId: PricingModelId.make("custom-model"),
      displayName: "Custom Model",
      inputPricePerMillion: 1.0,
      cachedInputPricePerMillion: Option.some(0.5),
      outputPricePerMillion: 2.0,
      currency: CurrencyCode.make("USD"),
      pricingSourceUrl: "https://example.com/pricing",
      lastVerified: "2025-01-01",
      effectiveDate: Option.none(),
      isEstimated: false,
    }

    expect(registry.has("custom-model")).toBe(false)
    registry.register(customEntry)
    expect(registry.has("custom-model")).toBe(true)

    const fetched = registry.get("custom-model")
    expect(Option.isSome(fetched)).toBe(true)
    if (Option.isSome(fetched)) {
      expect(fetched.value.displayName).toBe("Custom Model")
    }
  })

  it("merges user pricing overrides cleanly with base entries", () => {
    const base = defaultPricingRegistry.get("gpt-4o")
    expect(Option.isSome(base)).toBe(true)
    if (Option.isSome(base)) {
      const override: PricingOverride = {
        displayName: Option.some("Discounted GPT-4o"),
        inputPricePerMillion: 1.0,
        cachedInputPricePerMillion: Option.none(),
        outputPricePerMillion: 5.0,
        currency: Option.none(),
        pricingSourceUrl: Option.none(),
        isEstimated: Option.some(true),
      }

      const merged = applyPricingOverride(base.value, override)
      expect(merged.displayName).toBe("Discounted GPT-4o")
      expect(merged.inputPricePerMillion).toBe(1.0)
      expect(Option.isNone(merged.cachedInputPricePerMillion)).toBe(true)
      expect(merged.outputPricePerMillion).toBe(5.0)
      expect(merged.currency).toBe(base.value.currency)
      expect(merged.pricingSourceUrl).toBe(base.value.pricingSourceUrl)
      expect(merged.isEstimated).toBe(true)
    }
  })
})
