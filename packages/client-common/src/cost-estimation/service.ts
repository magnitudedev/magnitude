import { Context, Effect, Layer } from "effect"
import {
  AggregateCostBreakdown,
  CurrencyMismatchError,
  InvalidUsageDataError,
  ModelCostBreakdown,
  ModelUsageAggregationItem,
  ModelUsageTokens,
  PricingEntry,
  PricingResolution,
} from "./types"
import { calculateModelCost, aggregateMultiModelCost } from "./calculator"
import { resolveModelPricing, ResolvePricingOptions } from "./equivalence"
import { defaultPricingRegistry, PricingRegistry } from "./registry"

export interface CostEstimatorService {
  readonly registry: PricingRegistry
  readonly calculate: (
    usage: ModelUsageTokens,
    pricing: PricingEntry,
  ) => Effect.Effect<ModelCostBreakdown, InvalidUsageDataError>
  readonly resolve: (
    modelId: string,
    options?: ResolvePricingOptions,
  ) => Effect.Effect<PricingResolution>
  readonly aggregate: (
    items: ReadonlyArray<ModelUsageAggregationItem>,
    options?: Parameters<typeof aggregateMultiModelCost>[1],
  ) => Effect.Effect<AggregateCostBreakdown, CurrencyMismatchError | InvalidUsageDataError>
}

export class CostEstimator extends Context.Tag("@magnitudedev/client-common/CostEstimator")<
  CostEstimator,
  CostEstimatorService
>() {}

export const CostEstimatorLive = Layer.succeed(CostEstimator, {
  registry: defaultPricingRegistry,
  calculate: (usage, pricing) => calculateModelCost(usage, pricing),
  resolve: (modelId, options) => Effect.succeed(resolveModelPricing(modelId, options)),
  aggregate: (items, options) => aggregateMultiModelCost(items, options),
})
