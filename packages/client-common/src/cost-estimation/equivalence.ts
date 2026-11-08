import { Option } from "effect"
import { applyPricingOverride, defaultPricingRegistry, PricingRegistry } from "./registry"
import {
  CurrencyCode,
  PricingEntry,
  PricingModelId,
  PricingOverrides,
  PricingProviderId,
  PricingResolution,
} from "./types"

/**
 * Explicit mapping from local model identifiers (or base identifiers)
 * to hosted pricing models.
 *
 * DO NOT fuzzy match or guess. If a local model does not have a reliable,
 * explicit counterpart, it must remain Unmapped so the user can explicitly
 * select a comparison model or configure pricing.
 */
export const EXPLICIT_MODEL_EQUIVALENCE: ReadonlyMap<string, string> = new Map([
  // DeepSeek models
  ["deepseek-chat", "deepseek-chat"],
  ["deepseek-v3", "deepseek-chat"],
  ["deepseek-v4", "deepseek-chat"],
  ["deepseek-reasoner", "deepseek-reasoner"],
  ["deepseek-r1", "deepseek-reasoner"],
  ["deepseek-r1-distill-qwen-14b", "deepseek-reasoner"],
  ["deepseek-r1-distill-qwen-32b", "deepseek-reasoner"],
  ["deepseek-r1-distill-llama-70b", "deepseek-reasoner"],

  // Llama 3 models
  ["llama-3.3-70b-instruct", "llama-3.3-70b-instruct"],
  ["llama-3.3-70b", "llama-3.3-70b-instruct"],
  ["llama-3.1-70b-instruct", "llama-3.3-70b-instruct"],
  ["llama-3.1-70b", "llama-3.3-70b-instruct"],

  // Qwen 2.5 / 3.x models
  ["qwen-2.5-72b-instruct", "qwen-2.5-72b-instruct"],
  ["qwen2.5-72b-instruct", "qwen-2.5-72b-instruct"],
  ["qwen2.5-72b", "qwen-2.5-72b-instruct"],
  ["qwen-2.5-coder-32b", "qwen-2.5-coder-32b"],
  ["qwen2.5-coder-32b", "qwen-2.5-coder-32b"],
  ["qwen2.5-coder-32b-instruct", "qwen-2.5-coder-32b"],
  ["qwen2.5-coder-7b", "qwen-2.5-coder-32b"],
  ["qwen2.5-coder-7b-instruct", "qwen-2.5-coder-32b"],
  ["qwen3.5-122b", "qwen-2.5-72b-instruct"],
  ["qwen3.5-35b", "qwen-2.5-coder-32b"],
  ["qwen3.6-35b", "qwen-2.5-coder-32b"],
  ["qwen3.8-27b", "qwen-2.5-coder-32b"],
])

/**
 * Normalizes a local model ID to find its canonical base identifier.
 * Example:
 * - "llama-3.3-70b-instruct:gguf:q4" -> "llama-3.3-70b-instruct"
 * - "unsloth/Qwen3.5-122B-A10B-MTP-GGUF" -> "qwen3.5-122b"
 * - "hf:owner/repo:gguf:q4" -> "hf:owner/repo"
 */
export function extractBaseModelId(modelId: string): string {
  let cleaned = modelId.trim()

  // Handle hf: or local: scheme prefixes
  if (cleaned.startsWith("hf:")) {
    cleaned = cleaned.slice(3)
  } else if (cleaned.startsWith("local:")) {
    cleaned = cleaned.slice(6)
  }

  // Remove tag / format suffix e.g. :gguf:q4 or :gguf:q8_0
  const colonIndex = cleaned.indexOf(":")
  if (colonIndex > 0) {
    cleaned = cleaned.slice(0, colonIndex)
  }

  // Remove repo owner prefix e.g. unsloth/ or LiquidAI/ or user/
  const slashIndex = cleaned.lastIndexOf("/")
  if (slashIndex >= 0) {
    cleaned = cleaned.slice(slashIndex + 1)
  }

  // Remove trailing .gguf, -GGUF or _gguf
  cleaned = cleaned.replace(/[-_.]gguf$/i, "")

  // Remove known quant / arch markers
  cleaned = cleaned.replace(/[-_]qat.*$/i, "")
  cleaned = cleaned.replace(/[-_]dflash.*$/i, "")
  cleaned = cleaned.replace(/[-_]flash(-\d+)?$/i, "")
  cleaned = cleaned.replace(/[-_]mtp.*$/i, "")
  cleaned = cleaned.replace(/[-_]a\d+b.*$/i, "")

  return cleaned.toLowerCase()
}

export interface ResolvePricingOptions {
  /** Explicit comparison model selected by the user in the UI */
  readonly selectedComparisonModelId?: string
  /** User-defined pricing overrides */
  readonly overrides?: PricingOverrides
  /** Registry to resolve pricing from (defaults to built-in) */
  readonly registry?: PricingRegistry
}

/**
 * Resolves the appropriate hosted pricing for a local model.
 *
 * Resolution priority:
 * 1. User pricing override for this specific local modelId.
 * 2. User-selected comparison model.
 * 3. Direct match in pricing registry.
 * 4. Explicit model equivalence table match.
 * 5. Unmapped (no guess / no fabricated cost).
 */
export function resolveModelPricing(
  modelId: string,
  options: ResolvePricingOptions = {},
): PricingResolution {
  const registry = options.registry ?? defaultPricingRegistry

  // 1. Check user override for the exact model ID
  if (options.overrides && options.overrides[modelId]) {
    const override = options.overrides[modelId]!
    const baseEntry: PricingEntry = {
      providerId: PricingProviderId.make("custom"),
      modelId: PricingModelId.make(modelId),
      displayName: Option.getOrElse(override.displayName, () => modelId),
      inputPricePerMillion: override.inputPricePerMillion,
      cachedInputPricePerMillion: override.cachedInputPricePerMillion,
      outputPricePerMillion: override.outputPricePerMillion,
      currency: Option.getOrElse(override.currency, () => CurrencyCode.make("USD")),
      pricingSourceUrl: Option.getOrElse(override.pricingSourceUrl, () => "User configured override"),
      lastVerified: new Date().toISOString().slice(0, 10),
      effectiveDate: Option.none(),
      isEstimated: Option.getOrElse(override.isEstimated, () => false),
    }
    return {
      _tag: "Resolved",
      entry: baseEntry,
      source: "user_override",
    }
  }

  // 2. Check user-selected comparison model
  if (options.selectedComparisonModelId) {
    const selected = registry.get(options.selectedComparisonModelId)
    if (Option.isSome(selected)) {
      return {
        _tag: "Resolved",
        entry: selected.value,
        source: "user_selected",
      }
    }
    return {
      _tag: "MissingPricing",
      modelId: options.selectedComparisonModelId,
      reason: `Selected comparison model '${options.selectedComparisonModelId}' is not found in the pricing registry.`,
    }
  }

  // 3. Direct match in pricing registry
  const directMatch = registry.get(modelId)
  if (Option.isSome(directMatch)) {
    return {
      _tag: "Resolved",
      entry: directMatch.value,
      source: "direct",
    }
  }

  // 4. Explicit mapping table
  const baseId = extractBaseModelId(modelId)
  const mappedHostedId = EXPLICIT_MODEL_EQUIVALENCE.get(baseId) ?? EXPLICIT_MODEL_EQUIVALENCE.get(modelId)
  if (mappedHostedId) {
    const mappedEntry = registry.get(mappedHostedId)
    if (Option.isSome(mappedEntry)) {
      return {
        _tag: "Resolved",
        entry: mappedEntry.value,
        source: "explicit_mapping",
      }
    }
    return {
      _tag: "MissingPricing",
      modelId: mappedHostedId,
      reason: `Equivalent hosted model '${mappedHostedId}' is not found in the pricing registry.`,
    }
  }

  // 5. Unmapped - do not invent a cost
  return {
    _tag: "Unmapped",
    modelId,
    reason: `No hosted equivalent configured for '${modelId}'. Choose a comparison model to view API-equivalent cost.`,
  }
}
