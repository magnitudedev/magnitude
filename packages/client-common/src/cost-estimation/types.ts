import { Option, Schema } from "effect"

export const PricingProviderId = Schema.NonEmptyString.pipe(Schema.brand("PricingProviderId"))
export type PricingProviderId = typeof PricingProviderId.Type

export const PricingModelId = Schema.NonEmptyString.pipe(Schema.brand("PricingModelId"))
export type PricingModelId = typeof PricingModelId.Type

export const CurrencyCode = Schema.NonEmptyString.pipe(Schema.brand("CurrencyCode"))
export type CurrencyCode = typeof CurrencyCode.Type

const NonNegativeNumber = Schema.Number.pipe(Schema.nonNegative())
const NonNegativeInt = Schema.Number.pipe(Schema.int(), Schema.nonNegative())

/**
 * A published pricing entry for a hosted model.
 * All prices are in currency per million (1,000,000) tokens.
 */
export const PricingEntry = Schema.Struct({
  providerId: PricingProviderId,
  modelId: PricingModelId,
  displayName: Schema.NonEmptyString,
  inputPricePerMillion: NonNegativeNumber,
  cachedInputPricePerMillion: Schema.optionalWith(NonNegativeNumber, { as: "Option", exact: true }),
  outputPricePerMillion: NonNegativeNumber,
  currency: CurrencyCode,
  pricingSourceUrl: Schema.NonEmptyString,
  lastVerified: Schema.NonEmptyString,
  effectiveDate: Schema.optionalWith(Schema.NonEmptyString, { as: "Option", exact: true }),
  isEstimated: Schema.Boolean,
})
export type PricingEntry = typeof PricingEntry.Type

/**
 * Custom user-defined pricing override.
 */
export const PricingOverride = Schema.Struct({
  displayName: Schema.optionalWith(Schema.NonEmptyString, { as: "Option", exact: true }),
  inputPricePerMillion: NonNegativeNumber,
  cachedInputPricePerMillion: Schema.optionalWith(NonNegativeNumber, { as: "Option", exact: true }),
  outputPricePerMillion: NonNegativeNumber,
  currency: Schema.optionalWith(CurrencyCode, { as: "Option", exact: true }),
  pricingSourceUrl: Schema.optionalWith(Schema.NonEmptyString, { as: "Option", exact: true }),
  isEstimated: Schema.optionalWith(Schema.Boolean, { as: "Option", exact: true }),
})
export type PricingOverride = typeof PricingOverride.Type

export const PricingOverrides = Schema.Record({
  key: Schema.String,
  value: PricingOverride,
})
export type PricingOverrides = typeof PricingOverrides.Type

/**
 * Input token statistics for cost estimation.
 */
export const ModelUsageTokens = Schema.Struct({
  inputTokens: NonNegativeInt,
  cachedInputTokens: NonNegativeInt,
  outputTokens: NonNegativeInt,
})
export type ModelUsageTokens = typeof ModelUsageTokens.Type

/**
 * Deterministic cost breakdown for an inference workload.
 */
export const ModelCostBreakdown = Schema.Struct({
  uncachedInputCost: NonNegativeNumber,
  cachedInputCost: NonNegativeNumber,
  outputCost: NonNegativeNumber,
  totalCost: NonNegativeNumber,
  uncachedInputTokens: NonNegativeInt,
  cachedInputTokens: NonNegativeInt,
  outputTokens: NonNegativeInt,
  totalTokens: NonNegativeInt,
  cacheHitPercentage: NonNegativeNumber,
  cachedInputPricingAvailable: Schema.Boolean,
  effectiveCachedInputPricePerMillion: NonNegativeNumber,
  currency: CurrencyCode,
})
export type ModelCostBreakdown = typeof ModelCostBreakdown.Type

export type ResolutionSource = "direct" | "explicit_mapping" | "user_override" | "user_selected"

export type PricingResolution =
  | { readonly _tag: "Resolved"; readonly entry: PricingEntry; readonly source: ResolutionSource }
  | { readonly _tag: "Unmapped"; readonly modelId: string; readonly reason: string }
  | { readonly _tag: "MissingPricing"; readonly modelId: string; readonly reason: string }

export interface ModelUsageAggregationItem {
  readonly modelId: string
  readonly inputTokens: number
  readonly cachedInputTokens: number
  readonly outputTokens: number
  readonly requests?: number
}

export interface AggregateCostBreakdown {
  readonly uncachedInputCost: number
  readonly cachedInputCost: number
  readonly outputCost: number
  readonly totalCost: number
  readonly uncachedInputTokens: number
  readonly cachedInputTokens: number
  readonly outputTokens: number
  readonly totalTokens: number
  readonly cacheHitPercentage: number
  readonly currency: CurrencyCode
  readonly requests: number
  readonly modelsCalculated: ReadonlyArray<{
    readonly modelId: string
    readonly pricing: PricingEntry
    readonly breakdown: ModelCostBreakdown
  }>
  readonly modelsMissingPricing: ReadonlyArray<{
    readonly modelId: string
    readonly reason: string
  }>
  readonly isComplete: boolean
}

export interface DailyCostEstimate {
  readonly date: string
  readonly totalTokens: number
  readonly estimatedCost: number
  readonly currency: CurrencyCode
}

export class InvalidUsageDataError extends Schema.TaggedError<InvalidUsageDataError>()("InvalidUsageDataError", {
  message: Schema.String,
}) {}

export class MissingPricingError extends Schema.TaggedError<MissingPricingError>()("MissingPricingError", {
  modelId: Schema.String,
  message: Schema.String,
}) {}

export class CurrencyMismatchError extends Schema.TaggedError<CurrencyMismatchError>()("CurrencyMismatchError", {
  expected: Schema.String,
  received: Schema.String,
  message: Schema.String,
}) {}
