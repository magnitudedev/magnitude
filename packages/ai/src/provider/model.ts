import type { ProviderModelCapabilities } from "../model/capabilities"

/**
 * Reasoning effort levels — provider-agnostic.
 *
 * The provider's API accepts all five values. User-facing configuration
 * may restrict the subset (e.g. excluding "max" from the UI).
 */
export type ReasoningEffort = "none" | "low" | "medium" | "high" | "max"

/**
 * Pricing info for a provider model (per 1M tokens, in USD).
 */
export interface ModelPricingInfo {
  readonly input: number
  readonly output: number
  readonly cached_input: number | null
}

/**
 * Intrinsic model family capabilities — determined by the model architecture,
 * not by the provider serving it.
 *
 * Only includes properties that are truly invariant across providers.
 * - toolCalls: all models support tool calls — not a differentiating capability
 * - grammar: provider-level (depends on how the provider serves the model)
 * - reasoning: provider-level (effort options differ by provider, see ProviderModel.reasoningEfforts)
 */
export interface ModelFamilyCapabilities {
  readonly vision: boolean
}

/**
 * A distinct family of models that shares the same tokenizer and the same
 * intrinsic capabilities. One family may include multiple specific models
 * (e.g. glm-5.1 and glm-5.2 are the same family — same tokenizer,
 * same capabilities, just different versions).
 *
 * Properties here are invariant across all providers that serve models in
 * this family. Multiple ProviderModel entries can map to the same family.
 *
 * This is a provider-agnostic **interface** — it defines *what* a model
 * family is. The concrete `MODEL_FAMILIES` list and `classifyModelFamily`
 * classifier live in `packages/providers`.
 */
export interface ModelFamily {
  /** Family ID, e.g. "glm-5", "kimi-k2", "deepseek-v3" */
  readonly id: string
  /** Intrinsic capabilities — same for every model in this family */
  readonly capabilities: ModelFamilyCapabilities
}

/**
 * A model as offered by a specific provider.
 * Properties here MAY differ across providers serving the same family.
 */
export interface ProviderModel {
  /** Provider-specific model ID, e.g. "glm-5.2", "kimi-k2.7" */
  readonly providerModelId: string
  /** ID of the provider serving this model, e.g. "magnitude", "llamacpp" */
  readonly providerId: string
  /** Model family ID this provider model maps to (always populated) */
  readonly modelFamilyId: string
  /** Display name (may differ from family name) */
  readonly displayName: string
  /** Context window — provider-specific (can differ by provider) */
  readonly contextWindow: number
  /** Max output tokens — provider-specific */
  readonly maxOutputTokens: number
  /** Capabilities as served by this provider (may be a subset of family capabilities) */
  readonly capabilities: ProviderModelCapabilities
  /** Pricing — provider-specific (per 1M tokens, USD) */
  readonly pricing: ModelPricingInfo
  /**
   * Reasoning effort options available on this provider for this model.
   * Always has at least one entry (e.g. ["none"] for non-reasoning models).
   * Read from the provider at catalog time. Plain strings — the UI
   * capitalizes them for display.
   */
  readonly reasoningEfforts: readonly string[]
}

/** Re-exported for convenience. */
export type { ProviderModelCapabilities } from "../model/capabilities"
