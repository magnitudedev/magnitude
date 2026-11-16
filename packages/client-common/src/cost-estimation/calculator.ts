import { Either, Option } from "effect"
import {
  AggregateCostBreakdown,
  CurrencyCode,
  CurrencyMismatchError,
  DailyCostEstimate,
  InvalidUsageDataError,
  ModelCostBreakdown,
  ModelUsageAggregationItem,
  ModelUsageTokens,
  PricingEntry,
  PricingOverrides,
} from "./types"
import { defaultPricingRegistry, PricingRegistry } from "./registry"
import { resolveModelPricing } from "./equivalence"

/**
 * Rounds monetary amounts to 6 decimal places (micro-dollars)
 * to avoid floating point representation anomalies.
 */
export function roundCost(amount: number): number {
  return Math.round(amount * 1_000_000) / 1_000_000
}

/**
 * Core deterministic cost calculation for a single model workload.
 *
 * Formula:
 *   uncached_input = total_input - cached_input
 *   api_equivalent_cost = (
 *     uncached_input * input_price
 *     + cached_input * cached_input_price
 *     + output_tokens * output_price
 *   ) / 1_000_000
 *
 * Fallback policy:
 *   When cachedInputPricePerMillion is not specified (Option.none()),
 *   cached input tokens are charged at the standard input price (no synthetic discount).
 *
 * Invariant safety:
 *   - Cached input is clamped to total input so uncached tokens cannot be negative.
 *   - Negative token counts return an InvalidUsageDataError.
 *   - Zero tokens produce zero cost.
 */
export function calculateModelCost(
  usage: {
    readonly inputTokens: number
    readonly cachedInputTokens: number
    readonly outputTokens: number
  },
  pricing: PricingEntry,
): Either.Either<ModelCostBreakdown, InvalidUsageDataError> {
  // Validate non-negative tokens
  if (usage.inputTokens < 0 || usage.cachedInputTokens < 0 || usage.outputTokens < 0) {
    return Either.left(
      new InvalidUsageDataError({
        message: `Token counts cannot be negative: inputTokens=${usage.inputTokens}, cachedInputTokens=${usage.cachedInputTokens}, outputTokens=${usage.outputTokens}`,
      }),
    )
  }

  // Treat cached input as a subset of total input.
  // If reported cached tokens exceed total input, clamp to total input.
  const clampedCachedTokens = Math.min(usage.inputTokens, usage.cachedInputTokens)
  const uncachedInputTokens = usage.inputTokens - clampedCachedTokens
  const outputTokens = usage.outputTokens
  const totalTokens = usage.inputTokens + outputTokens

  // Fallback policy for missing cached-input pricing:
  // If unavailable, cached tokens are charged at the standard input rate.
  const cachedPricingAvailable = Option.isSome(pricing.cachedInputPricePerMillion)
  const effectiveCachedPrice = Option.getOrElse(
    pricing.cachedInputPricePerMillion,
    () => pricing.inputPricePerMillion,
  )

  const uncachedInputCost = roundCost((uncachedInputTokens * pricing.inputPricePerMillion) / 1_000_000)
  const cachedInputCost = roundCost((clampedCachedTokens * effectiveCachedPrice) / 1_000_000)
  const outputCost = roundCost((outputTokens * pricing.outputPricePerMillion) / 1_000_000)
  const totalCost = roundCost(uncachedInputCost + cachedInputCost + outputCost)

  const cacheHitPercentage = usage.inputTokens > 0
    ? roundCost((clampedCachedTokens / usage.inputTokens) * 100)
    : 0

  return Either.right({
    uncachedInputCost,
    cachedInputCost,
    outputCost,
    totalCost,
    uncachedInputTokens,
    cachedInputTokens: clampedCachedTokens,
    outputTokens,
    totalTokens,
    cacheHitPercentage,
    cachedInputPricingAvailable: cachedPricingAvailable,
    effectiveCachedInputPricePerMillion: effectiveCachedPrice,
    currency: pricing.currency,
  })
}

/**
 * Aggregates API-equivalent costs across multiple models.
 *
 * Verifies that all calculated models share the same currency.
 * If models have conflicting currencies, returns a CurrencyMismatchError.
 */
export function aggregateMultiModelCost(
  items: ReadonlyArray<ModelUsageAggregationItem>,
  options: {
    readonly registry?: PricingRegistry
    readonly overrides?: PricingOverrides
    readonly targetCurrency?: CurrencyCode
  } = {},
): Either.Either<AggregateCostBreakdown, CurrencyMismatchError | InvalidUsageDataError> {
  const registry = options.registry ?? defaultPricingRegistry
  let uncachedInputCost = 0
  let cachedInputCost = 0
  let outputCost = 0
  let totalCost = 0
  let uncachedInputTokens = 0
  let cachedInputTokens = 0
  let outputTokens = 0
  let totalTokens = 0
  let totalRequests = 0
  let activeCurrency: CurrencyCode = options.targetCurrency ?? CurrencyCode.make("USD")

  const modelsCalculated: Array<{
    readonly modelId: string
    readonly pricing: PricingEntry
    readonly breakdown: ModelCostBreakdown
  }> = []

  const modelsMissingPricing: Array<{
    readonly modelId: string
    readonly reason: string
  }> = []

  for (const item of items) {
    totalRequests += item.requests ?? 1

    const resolution = resolveModelPricing(item.modelId, {
      registry,
      overrides: options.overrides,
    })

    if (resolution._tag !== "Resolved") {
      modelsMissingPricing.push({
        modelId: item.modelId,
        reason: resolution.reason,
      })
      continue
    }

    const pricing = resolution.entry

    // Check currency compatibility
    if (modelsCalculated.length === 0) {
      activeCurrency = pricing.currency
    } else if (pricing.currency !== activeCurrency) {
      return Either.left(
        new CurrencyMismatchError({
          expected: activeCurrency,
          received: pricing.currency,
          message: `Cannot aggregate models with different currencies: ${activeCurrency} vs ${pricing.currency} for model '${item.modelId}'.`,
        }),
      )
    }

    const costEither = calculateModelCost(item, pricing)
    if (Either.isLeft(costEither)) {
      return Either.left(costEither.left)
    }

    const breakdown = costEither.right
    uncachedInputCost += breakdown.uncachedInputCost
    cachedInputCost += breakdown.cachedInputCost
    outputCost += breakdown.outputCost
    totalCost += breakdown.totalCost
    uncachedInputTokens += breakdown.uncachedInputTokens
    cachedInputTokens += breakdown.cachedInputTokens
    outputTokens += breakdown.outputTokens
    totalTokens += breakdown.totalTokens

    modelsCalculated.push({
      modelId: item.modelId,
      pricing,
      breakdown,
    })
  }

  const allInputTokens = uncachedInputTokens + cachedInputTokens
  const cacheHitPercentage = allInputTokens > 0
    ? roundCost((cachedInputTokens / allInputTokens) * 100)
    : 0

  return Either.right({
    uncachedInputCost: roundCost(uncachedInputCost),
    cachedInputCost: roundCost(cachedInputCost),
    outputCost: roundCost(outputCost),
    totalCost: roundCost(totalCost),
    uncachedInputTokens,
    cachedInputTokens,
    outputTokens,
    totalTokens,
    cacheHitPercentage,
    currency: activeCurrency,
    requests: totalRequests,
    modelsCalculated,
    modelsMissingPricing,
    isComplete: modelsMissingPricing.length === 0,
  })
}

/**
 * Projects API-equivalent cost over historical daily token activity.
 *
 * For each calendar day in dailyActivity, calculates the proportional
 * estimated cost based on that day's total tokens relative to the total tokens.
 */
export function estimateHistoricalDailyCosts(
  dailyActivity: ReadonlyArray<{ readonly date: string; readonly totalTokens: number }>,
  totalTokens: number,
  totalCost: number,
  currency: CurrencyCode = CurrencyCode.make("USD"),
): ReadonlyArray<DailyCostEstimate> {
  if (totalTokens <= 0 || totalCost <= 0) {
    return dailyActivity.map(day => ({
      date: day.date,
      totalTokens: day.totalTokens,
      estimatedCost: 0,
      currency,
    }))
  }

  const effectiveRatePerToken = totalCost / totalTokens

  return dailyActivity.map(day => ({
    date: day.date,
    totalTokens: day.totalTokens,
    estimatedCost: roundCost(day.totalTokens * effectiveRatePerToken),
    currency,
  }))
}
