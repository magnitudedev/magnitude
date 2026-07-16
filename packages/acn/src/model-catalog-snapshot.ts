import {
  ProviderCatalogStale,
  ProviderCatalogUnavailable,
  type ProviderCatalogFailure,
  type ProviderCatalogOutcome,
  type ProviderId,
  type ProviderModel,
} from "@magnitudedev/sdk"

/**
 * Apply a partially successful aggregate refresh without dropping models from
 * providers that failed. Successful providers remain authoritative, including
 * an authoritative empty result.
 */
export interface FoldedProviderCatalogs {
  readonly byProvider: ReadonlyMap<ProviderId, readonly ProviderModel[]>
  readonly failuresByProvider: ReadonlyMap<ProviderId, ProviderCatalogFailure>
  readonly models: readonly ProviderModel[]
  readonly failures: readonly ProviderCatalogFailure[]
}

export function foldProviderCatalogOutcomes(
  previous: Pick<FoldedProviderCatalogs, "byProvider" | "failuresByProvider">,
  outcomes: readonly ProviderCatalogOutcome[],
): FoldedProviderCatalogs {
  const byProvider = new Map(previous.byProvider)
  const failuresByProvider = new Map(previous.failuresByProvider)

  for (const outcome of outcomes) {
    if (outcome._tag === "Success") {
      byProvider.set(outcome.providerId, outcome.models)
      failuresByProvider.delete(outcome.providerId)
      continue
    }
    const retained = byProvider.get(outcome.providerId) ?? []
    failuresByProvider.set(outcome.providerId, retained.length > 0
      ? new ProviderCatalogStale({ providerId: outcome.providerId, message: outcome.failure.message })
      : new ProviderCatalogUnavailable({ providerId: outcome.providerId, message: outcome.failure.message }))
  }

  return {
    byProvider,
    failuresByProvider,
    models: [...byProvider.values()].flat(),
    failures: [...failuresByProvider.values()],
  }
}
