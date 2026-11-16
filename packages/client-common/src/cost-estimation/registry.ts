import { Option } from "effect"
import {
  CurrencyCode,
  PricingEntry,
  PricingModelId,
  PricingOverride,
  PricingProviderId,
} from "./types"

const USD = CurrencyCode.make("USD")

/**
 * Verified pricing entries from official provider documentation.
 * Last verified: January 2025.
 * Currency: USD per million tokens.
 */
export const BUILT_IN_PRICING: readonly PricingEntry[] = [
  // Anthropic
  {
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
  },
  {
    providerId: PricingProviderId.make("anthropic"),
    modelId: PricingModelId.make("claude-3-5-haiku-20241022"),
    displayName: "Claude 3.5 Haiku",
    inputPricePerMillion: 0.8,
    cachedInputPricePerMillion: Option.some(0.08),
    outputPricePerMillion: 4.0,
    currency: USD,
    pricingSourceUrl: "https://www.anthropic.com/pricing",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },

  // OpenAI
  {
    providerId: PricingProviderId.make("openai"),
    modelId: PricingModelId.make("gpt-4o"),
    displayName: "GPT-4o",
    inputPricePerMillion: 2.5,
    cachedInputPricePerMillion: Option.some(1.25),
    outputPricePerMillion: 10.0,
    currency: USD,
    pricingSourceUrl: "https://openai.com/api/pricing/",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },
  {
    providerId: PricingProviderId.make("openai"),
    modelId: PricingModelId.make("gpt-4o-mini"),
    displayName: "GPT-4o mini",
    inputPricePerMillion: 0.15,
    cachedInputPricePerMillion: Option.some(0.075),
    outputPricePerMillion: 0.6,
    currency: USD,
    pricingSourceUrl: "https://openai.com/api/pricing/",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },

  // Google
  {
    providerId: PricingProviderId.make("google"),
    modelId: PricingModelId.make("gemini-1.5-pro"),
    displayName: "Gemini 1.5 Pro",
    inputPricePerMillion: 1.25,
    cachedInputPricePerMillion: Option.some(0.3125),
    outputPricePerMillion: 5.0,
    currency: USD,
    pricingSourceUrl: "https://ai.google.dev/pricing",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },
  {
    providerId: PricingProviderId.make("google"),
    modelId: PricingModelId.make("gemini-1.5-flash"),
    displayName: "Gemini 1.5 Flash",
    inputPricePerMillion: 0.075,
    cachedInputPricePerMillion: Option.some(0.01875),
    outputPricePerMillion: 0.3,
    currency: USD,
    pricingSourceUrl: "https://ai.google.dev/pricing",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },

  // DeepSeek
  {
    providerId: PricingProviderId.make("deepseek"),
    modelId: PricingModelId.make("deepseek-chat"),
    displayName: "DeepSeek-V3",
    inputPricePerMillion: 0.14,
    cachedInputPricePerMillion: Option.some(0.014),
    outputPricePerMillion: 0.28,
    currency: USD,
    pricingSourceUrl: "https://api-docs.deepseek.com/quick_start/pricing",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },
  {
    providerId: PricingProviderId.make("deepseek"),
    modelId: PricingModelId.make("deepseek-reasoner"),
    displayName: "DeepSeek-R1",
    inputPricePerMillion: 0.55,
    cachedInputPricePerMillion: Option.some(0.14),
    outputPricePerMillion: 2.19,
    currency: USD,
    pricingSourceUrl: "https://api-docs.deepseek.com/quick_start/pricing",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },

  // Hosted open-weight models (Together AI / Alibaba Cloud)
  // Models with no cached input discount use cachedInputPricePerMillion: None
  {
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
  },
  {
    providerId: PricingProviderId.make("alibaba-cloud"),
    modelId: PricingModelId.make("qwen-2.5-72b-instruct"),
    displayName: "Qwen 2.5 72B Instruct (Hosted)",
    inputPricePerMillion: 0.35,
    cachedInputPricePerMillion: Option.none(),
    outputPricePerMillion: 0.4,
    currency: USD,
    pricingSourceUrl: "https://www.alibabacloud.com/help/en/model-studio/developer-reference/model-square/",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },
  {
    providerId: PricingProviderId.make("alibaba-cloud"),
    modelId: PricingModelId.make("qwen-2.5-coder-32b"),
    displayName: "Qwen 2.5 Coder 32B (Hosted)",
    inputPricePerMillion: 0.2,
    cachedInputPricePerMillion: Option.none(),
    outputPricePerMillion: 0.2,
    currency: USD,
    pricingSourceUrl: "https://www.alibabacloud.com/help/en/model-studio/developer-reference/model-square/",
    lastVerified: "2025-01-15",
    effectiveDate: Option.none(),
    isEstimated: false,
  },
]

export class PricingRegistry {
  private readonly entries = new Map<string, PricingEntry>()

  constructor(initialEntries: readonly PricingEntry[] = BUILT_IN_PRICING) {
    for (const entry of initialEntries) {
      this.entries.set(entry.modelId, entry)
    }
  }

  get(modelId: string): Option.Option<PricingEntry> {
    return Option.fromNullable(this.entries.get(modelId))
  }

  has(modelId: string): boolean {
    return this.entries.has(modelId)
  }

  register(entry: PricingEntry): void {
    this.entries.set(entry.modelId, entry)
  }

  list(): readonly PricingEntry[] {
    return Array.from(this.entries.values())
  }
}

export const defaultPricingRegistry = new PricingRegistry()

/**
 * Merges a base pricing entry with a user override.
 */
export function applyPricingOverride(base: PricingEntry, override: PricingOverride): PricingEntry {
  return {
    providerId: base.providerId,
    modelId: base.modelId,
    displayName: Option.getOrElse(override.displayName, () => base.displayName),
    inputPricePerMillion: override.inputPricePerMillion,
    cachedInputPricePerMillion: override.cachedInputPricePerMillion,
    outputPricePerMillion: override.outputPricePerMillion,
    currency: Option.getOrElse(override.currency, () => base.currency),
    pricingSourceUrl: Option.getOrElse(override.pricingSourceUrl, () => base.pricingSourceUrl),
    lastVerified: base.lastVerified,
    effectiveDate: base.effectiveDate,
    isEstimated: Option.getOrElse(override.isEstimated, () => false),
  }
}
