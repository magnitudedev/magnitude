import { useState, useId } from "react"
import { Option, Either } from "effect"
import type { ServingUsageSnapshot } from "@magnitudedev/sdk"
import {
  calculateModelCost,
  defaultPricingRegistry,
  resolveModelPricing,
  type PricingEntry,
  type ModelCostBreakdown,
} from "@magnitudedev/client-common"
import {
  InfoIcon,
  ArrowDownIcon,
  ArrowUpIcon,
  StackIcon,
  ArrowUpRightIcon,
  ReceiptIcon,
  CheckCircleIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select"
import { ActionTooltip } from "./ui/tooltip"
import { SkeletonLine } from "./page-skeletons"
import { pageLayout } from "./page-layout"

const formatNumber = (value: number) => value.toLocaleString()
const formatDecimal = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 1 })

export function formatCostCurrency(amount: number, currency = "USD"): string {
  if (amount === 0) return "$0.00"
  if (amount < 0.01) {
    return `< $0.01`
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount)
}

export function formatExactCost(amount: number, currency = "USD"): string {
  if (amount === 0) return "$0.0000"
  if (amount < 0.0001) {
    return `< $0.0001`
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  }).format(amount)
}

export interface ApiEquivalentCostProps {
  readonly usage: Extract<ServingUsageSnapshot, { _tag: "Available" }> | null
  readonly selectedModelId: string
  readonly selectedModelDisplayName?: string
  readonly period: "Today" | "AllTime"
}

export function ApiEquivalentCost({
  usage,
  selectedModelId,
  selectedModelDisplayName,
  period,
}: ApiEquivalentCostProps) {
  const [comparisonModelId, setComparisonModelId] = useState<string>("auto")
  const [showAssumptions, setShowAssumptions] = useState<boolean>(false)
  const assumptionsId = useId()

  const availableComparisonModels = defaultPricingRegistry.list()

  // Resolve pricing
  const effectiveModelId = selectedModelId || "claude-3-5-sonnet-20241022" // Baseline comparison for "All models"
  const isAllModels = !selectedModelId

  const resolution = resolveModelPricing(effectiveModelId, {
    selectedComparisonModelId: comparisonModelId !== "auto" ? comparisonModelId : undefined,
  })

  let costBreakdown: ModelCostBreakdown | null = null
  let pricingEntry: PricingEntry | null = null

  if (usage && resolution._tag === "Resolved") {
    pricingEntry = resolution.entry
    const costEither = calculateModelCost(
      {
        inputTokens: usage.inputTokens,
        cachedInputTokens: usage.cachedInputTokens,
        outputTokens: usage.outputTokens,
      },
      pricingEntry,
    )
    if (Either.isRight(costEither)) {
      costBreakdown = costEither.right
    }
  }

  const missingCache = usage ? usage.requests - usage.cachedInputRequests : 0
  const disclaimerText =
    "API-equivalent cost estimates what your local inference usage would cost at the selected hosted provider's published token rates. This is not your actual expenditure or net savings."

  return (
    <section
      aria-label="API-equivalent cost"
      className="rounded-2xl border border-slate-200 bg-white p-6 dark:border-slate-750 dark:bg-slate-850"
    >
      {/* Header and comparison model selector */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4 dark:border-slate-750">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium">API-Equivalent Cost</h2>
          <ActionTooltip
            label={disclaimerText}
            trigger={
              <button
                type="button"
                aria-label={disclaimerText}
                className="inline-flex rounded-sm text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              >
                <InfoIcon className="size-4" />
              </button>
            }
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="comparison-model-select" className="text-xs text-slate-500">
            Compare against:
          </label>
          <Select value={comparisonModelId} onValueChange={value => setComparisonModelId(value ?? "auto")}>
            <SelectTrigger id="comparison-model-select" aria-label="Comparison hosted model" className="w-[260px] max-w-full">
              <SelectValue>
                {comparisonModelId === "auto"
                  ? resolution._tag === "Resolved"
                    ? `Auto (${resolution.entry.displayName})`
                    : "Auto (Choose comparison)"
                  : availableComparisonModels.find(m => m.modelId === comparisonModelId)?.displayName ?? comparisonModelId}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">
                {resolution._tag === "Resolved"
                  ? `Auto (${resolution.entry.displayName})`
                  : "Auto (Choose comparison)"}
              </SelectItem>
              {availableComparisonModels.map(item => (
                <SelectItem key={item.modelId} value={item.modelId}>
                  {item.displayName} ({item.providerId})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Main cost summary */}
      <div className="mt-5">
        {usage === null ? (
          // Loading skeleton state
          <div className="space-y-4">
            <div className="flex items-baseline justify-between">
              <SkeletonLine className="h-9 text-3xl" width="160px" />
              <SkeletonLine className="h-5 text-sm" width="120px" />
            </div>
            <div className={pageLayout.usageCost}>
              <SkeletonLine className="h-8 text-2xl" width="100px" />
              <SkeletonLine className="h-8 text-2xl" width="100px" />
              <SkeletonLine className="h-8 text-2xl" width="100px" />
            </div>
          </div>
        ) : resolution._tag === "Unmapped" ? (
          // Unmapped model state - never fabricate 0!
          <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4 dark:border-amber-900/40 dark:bg-amber-950/20">
            <div className="flex items-start gap-3">
              <WarningCircleIcon className="mt-0.5 size-5 shrink-0 text-amber-600 dark:text-amber-400" />
              <div>
                <p className="text-sm font-medium text-amber-900 dark:text-amber-200">
                  No default hosted equivalent for this model
                </p>
                <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">
                  {selectedModelDisplayName ?? selectedModelId} does not have an automatic hosted equivalent.
                  Select a comparable model from the dropdown above to estimate what your workload would have cost.
                </p>
              </div>
            </div>
          </div>
        ) : costBreakdown ? (
          // Calculated cost display
          <div>
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div>
                <p className="text-sm text-slate-500">
                  Estimated cost ({period === "Today" ? "Today" : "All time"})
                </p>
                <div className="mt-3 flex items-baseline gap-3">
                  <p className="font-heading text-3xl tabular-nums text-slate-900 dark:text-slate-100" data-cost="total">
                    {formatCostCurrency(costBreakdown.totalCost, costBreakdown.currency)}
                  </p>
                  {costBreakdown.totalCost > 0 && costBreakdown.totalCost < 0.01 && (
                    <span className="text-xs text-slate-500 tabular-nums">
                      ({formatExactCost(costBreakdown.totalCost, costBreakdown.currency)})
                    </span>
                  )}
                  {pricingEntry && (
                    <span
                      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        pricingEntry.isEstimated
                          ? "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"
                          : "bg-blue-50 text-blue-700 dark:bg-blue-950/50 dark:text-blue-300"
                      }`}
                    >
                      {pricingEntry.isEstimated ? (
                        <>Estimated pricing</>
                      ) : (
                        <>
                          <CheckCircleIcon className="size-3" />
                          Published rates
                        </>
                      )}
                    </span>
                  )}
                </div>
              </div>

              <button
                type="button"
                onClick={() => setShowAssumptions(prev => !prev)}
                aria-expanded={showAssumptions}
                aria-controls={assumptionsId}
                className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
              >
                <ReceiptIcon className="size-3.5" />
                {showAssumptions ? "Hide pricing details" : "View pricing assumptions"}
              </button>
            </div>

            {/* Pricing Assumptions Expandable Panel */}
            {showAssumptions && pricingEntry && (
              <div
                id={assumptionsId}
                className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs dark:border-slate-750 dark:bg-slate-900/50"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-2 dark:border-slate-750">
                  <span className="font-medium text-slate-700 dark:text-slate-300">
                    Pricing Basis: {pricingEntry.displayName} ({pricingEntry.providerId})
                  </span>
                  <a
                    href={pricingEntry.pricingSourceUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-blue-600 hover:underline dark:text-blue-400"
                  >
                    Provider pricing source
                    <ArrowUpRightIcon className="size-3" />
                  </a>
                </div>
                <div className="mt-2.5 grid grid-cols-1 gap-2 sm:grid-cols-3">
                  <div>
                    <span className="text-slate-500">Standard input:</span>{" "}
                    <span className="font-medium tabular-nums text-slate-800 dark:text-slate-200">
                      ${pricingEntry.inputPricePerMillion} / 1M tokens
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Cached input:</span>{" "}
                    <span className="font-medium tabular-nums text-slate-800 dark:text-slate-200">
                      {Option.isSome(pricingEntry.cachedInputPricePerMillion)
                        ? `$${pricingEntry.cachedInputPricePerMillion.value} / 1M tokens`
                        : `No cache discount ($${pricingEntry.inputPricePerMillion}/1M)`}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Output:</span>{" "}
                    <span className="font-medium tabular-nums text-slate-800 dark:text-slate-200">
                      ${pricingEntry.outputPricePerMillion} / 1M tokens
                    </span>
                  </div>
                </div>
                <div className="mt-2 text-slate-500">
                  Rates verified as of {pricingEntry.lastVerified}. Currency: {pricingEntry.currency}.
                </div>
              </div>
            )}

            {/* Cost Breakdown Grid */}
            <div className={`mt-6 ${pageLayout.usageCost}`}>
              <div className="min-w-0">
                <p className="flex items-center gap-2 text-sm text-slate-500">
                  <ArrowDownIcon className="size-4" />
                  Uncached input cost
                </p>
                <p className="mt-4 text-2xl tabular-nums" data-cost="uncached-input">
                  {formatCostCurrency(costBreakdown.uncachedInputCost, costBreakdown.currency)}
                </p>
                <p className="mt-1 text-xs text-slate-500 tabular-nums">
                  {formatNumber(costBreakdown.uncachedInputTokens)} tokens
                </p>
              </div>

              <div className="min-w-0">
                <p className="flex items-center gap-2 text-sm text-slate-500">
                  <StackIcon className="size-4" />
                  Cached input cost
                  {pricingEntry && !Option.isSome(pricingEntry.cachedInputPricePerMillion) && (
                    <ActionTooltip
                      label="This provider does not publish prompt cache discounts. Cached tokens are charged at standard input rate."
                      trigger={
                        <button type="button" aria-label="No cache discount" className="text-slate-400">
                          <InfoIcon className="size-3.5" />
                        </button>
                      }
                    />
                  )}
                </p>
                <p className="mt-4 text-2xl tabular-nums" data-cost="cached-input">
                  {formatCostCurrency(costBreakdown.cachedInputCost, costBreakdown.currency)}
                </p>
                <p className="mt-1 text-xs text-slate-500 tabular-nums">
                  {formatNumber(costBreakdown.cachedInputTokens)} tokens
                  {costBreakdown.cacheHitPercentage > 0 && ` (${formatDecimal(costBreakdown.cacheHitPercentage)}% hit)`}
                </p>
              </div>

              <div className="min-w-0">
                <p className="flex items-center gap-2 text-sm text-slate-500">
                  <ArrowUpIcon className="size-4" />
                  Output cost
                </p>
                <p className="mt-4 text-2xl tabular-nums" data-cost="output">
                  {formatCostCurrency(costBreakdown.outputCost, costBreakdown.currency)}
                </p>
                <p className="mt-1 text-xs text-slate-500 tabular-nums">
                  {formatNumber(costBreakdown.outputTokens)} tokens
                </p>
              </div>
            </div>

            {/* Note on missing cache evidence if present */}
            {missingCache > 0 && (
              <p className="mt-3 text-xs text-slate-500">
                Note: Prompt cache usage was not reported for {formatNumber(missingCache)} requests.
                Unreported tokens are evaluated at standard input rates.
              </p>
            )}

            {/* Explanatory footer disclaimer */}
            <p className="mt-4 border-t border-slate-200 pt-3 text-xs text-slate-500 dark:border-slate-750">
              {disclaimerText}
            </p>
          </div>
        ) : null}
      </div>
    </section>
  )
}
