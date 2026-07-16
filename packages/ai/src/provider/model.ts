import { Schema } from "effect"
import { ProviderModelCapabilitiesSchema, type ProviderModelCapabilities } from "../model/capabilities"

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
const FiniteNonNegative = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
const PositiveSafeInteger = Schema.Number.pipe(Schema.int(), Schema.positive(), Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER))
export const ProviderIdSchema = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(256), Schema.brand("ProviderId"))
export type ProviderId = Schema.Schema.Type<typeof ProviderIdSchema>
export const ProviderModelIdSchema = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(4096), Schema.brand("ProviderModelId"))
export type ProviderModelId = Schema.Schema.Type<typeof ProviderModelIdSchema>
export const ModelFamilyIdSchema = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(512), Schema.brand("ModelFamilyId"))
export type ModelFamilyId = Schema.Schema.Type<typeof ModelFamilyIdSchema>
export const ModelPricingInfoSchema = Schema.Struct({
  input: FiniteNonNegative,
  output: FiniteNonNegative,
  cached_input: Schema.NullOr(FiniteNonNegative),
})
export type ModelPricingInfo = Schema.Schema.Type<typeof ModelPricingInfoSchema>

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

export const ProviderModelDisabledReasonSchema = Schema.Literal(
  "insufficient_resources", "provider_unavailable", "model_unavailable", "incompatible_runtime", "invalid_configuration",
)
export type ProviderModelDisabledReason = Schema.Schema.Type<typeof ProviderModelDisabledReasonSchema>
export const ProviderModelAvailabilitySchema = Schema.Union(
  Schema.TaggedStruct("Available", {}),
  Schema.TaggedStruct("Disabled", { reason: ProviderModelDisabledReasonSchema }),
)
export type ProviderModelAvailability = Schema.Schema.Type<typeof ProviderModelAvailabilitySchema>

export const AVAILABLE_PROVIDER_MODEL: ProviderModelAvailability = { _tag: "Available" }

export const isProviderModelAvailable = (
  model: Pick<ProviderModel, "availability">,
): boolean => model.availability._tag === "Available"

/**
 * A model as offered by a specific provider.
 * Properties here MAY differ across providers serving the same family.
 */
export const ProviderModelSchema = Schema.Struct({
  providerModelId: Schema.String,
  providerId: Schema.String,
  modelFamilyId: Schema.optional(Schema.String),
  displayName: Schema.String.pipe(Schema.minLength(1)),
  contextWindow: PositiveSafeInteger,
  maxOutputTokens: PositiveSafeInteger,
  capabilities: ProviderModelCapabilitiesSchema,
  availability: ProviderModelAvailabilitySchema,
  pricing: ModelPricingInfoSchema,
  reasoningEfforts: Schema.Array(Schema.String.pipe(Schema.minLength(1))).pipe(Schema.minItems(1)),
})
export type ProviderModel = Schema.Schema.Type<typeof ProviderModelSchema>

/** Re-exported for convenience. */
export type { ProviderModelCapabilities } from "../model/capabilities"
