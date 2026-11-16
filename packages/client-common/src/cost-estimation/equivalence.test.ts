import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  EXPLICIT_MODEL_EQUIVALENCE,
  extractBaseModelId,
  resolveModelPricing,
} from "./equivalence"
import { PricingRegistry } from "./registry"
import { CurrencyCode, PricingEntry, PricingModelId, PricingProviderId } from "./types"

describe("Model Equivalence and Resolution", () => {
  it("normalizes diverse model strings into canonical base identifiers", () => {
    expect(extractBaseModelId("llama-3.3-70b-instruct:gguf:q4_k_m")).toBe("llama-3.3-70b-instruct")
    expect(extractBaseModelId("unsloth/DeepSeek-V4-Flash-0731-GGUF")).toBe("deepseek-v4")
    expect(extractBaseModelId("magnitudedev/Qwen3.5-122B-A10B-MTP-GGUF")).toBe("qwen3.5-122b")
    expect(extractBaseModelId("LiquidAI/LFM2.5-2.6B-GGUF")).toBe("lfm2.5-2.6b")
    expect(extractBaseModelId("hf:user/my-custom-model.gguf")).toBe("my-custom-model")
  })

  it("resolves explicitly mapped models", () => {
    const res = resolveModelPricing("deepseek-r1-distill-qwen-14b:gguf:q4")
    expect(res._tag).toBe("Resolved")
    if (res._tag === "Resolved") {
      expect(res.source).toBe("explicit_mapping")
      expect(res.entry.modelId).toBe("deepseek-reasoner")
      expect(res.entry.displayName).toBe("DeepSeek-R1")
    }
  })

  it("does not fuzzy match unrelated models", () => {
    // A completely unrelated model like LFM or MiniCPM should NOT map to Sonnet or GPT-4o
    const res = resolveModelPricing("LiquidAI/LFM2.5-2.6B-GGUF")
    expect(res._tag).toBe("Unmapped")
    if (res._tag === "Unmapped") {
      expect(res.reason).toContain("No hosted equivalent configured")
    }
  })

  it("prioritizes user pricing overrides over explicit mapping", () => {
    const res = resolveModelPricing("llama-3.3-70b-instruct", {
      overrides: {
        "llama-3.3-70b-instruct": {
          displayName: Option.some("Custom Llama Pricing"),
          inputPricePerMillion: 0.5,
          cachedInputPricePerMillion: Option.some(0.1),
          outputPricePerMillion: 1.0,
          currency: Option.some(CurrencyCode.make("USD")),
          pricingSourceUrl: Option.none(),
          isEstimated: Option.some(true),
        },
      },
    })
    expect(res._tag).toBe("Resolved")
    if (res._tag === "Resolved") {
      expect(res.source).toBe("user_override")
      expect(res.entry.displayName).toBe("Custom Llama Pricing")
      expect(res.entry.inputPricePerMillion).toBe(0.5)
      expect(res.entry.outputPricePerMillion).toBe(1.0)
    }
  })

  it("allows user to explicitly pick a comparison model", () => {
    const res = resolveModelPricing("LiquidAI/LFM2.5-2.6B-GGUF", {
      selectedComparisonModelId: "claude-3-5-sonnet-20241022",
    })
    expect(res._tag).toBe("Resolved")
    if (res._tag === "Resolved") {
      expect(res.source).toBe("user_selected")
      expect(res.entry.displayName).toBe("Claude 3.5 Sonnet")
    }
  })

  it("returns MissingPricing when selected comparison model does not exist", () => {
    const res = resolveModelPricing("LiquidAI/LFM2.5-2.6B-GGUF", {
      selectedComparisonModelId: "non-existent-hosted-model",
    })
    expect(res._tag).toBe("MissingPricing")
  })
})
