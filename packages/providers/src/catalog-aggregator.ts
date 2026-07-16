import { Effect } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import {
  ModelCatalogError,
  type ModelCatalog,
  type ProviderModel,
  type ProviderId,
  type ModelFamilyId,
} from "@magnitudedev/ai"

/**
 * Merge multiple provider catalogs into a single aggregated catalog.
 * The aggregated catalog type-erases to ProviderModel (base type).
 */
export function makeAggregatedCatalog(
  providers: readonly { readonly id: ProviderId; readonly catalog: ModelCatalog<any> }[],
): ModelCatalog<ProviderModel> {
  const collect = (operation: "list" | "refresh"): ModelCatalog<ProviderModel>["list"] =>
    inspectProviderCatalogs(providers, operation).pipe(
      Effect.map((outcomes) => outcomes.flatMap((outcome) => outcome._tag === "Success" ? outcome.models : [])),
    )

  const list = collect("list")

  const get: ModelCatalog<ProviderModel>["get"] = (providerId, providerModelId) =>
    Effect.gen(function* () {
      const provider = providers.find((p) => p.id === providerId)
      if (!provider) return yield* new ModelCatalogError({ message: `Unknown provider: ${providerId}` })
      return yield* provider.catalog.get(providerId, providerModelId) as Effect.Effect<ProviderModel, ModelCatalogError, HttpClient.HttpClient>
    })

  const refresh = collect("refresh")

  return { list, get, refresh }
}

export type ProviderCatalogOutcome =
  | {
      readonly _tag: "Success"
      readonly providerId: ProviderId
      readonly models: readonly ProviderModel[]
    }
  | {
      readonly _tag: "Failure"
      readonly providerId: ProviderId
      readonly failure: ModelCatalogError
    }

export const inspectProviderCatalogs = (
  providers: readonly { readonly id: ProviderId; readonly catalog: ModelCatalog<any> }[],
  operation: "list" | "refresh",
): Effect.Effect<readonly ProviderCatalogOutcome[], never, HttpClient.HttpClient> => Effect.gen(function* () {
    const results = yield* Effect.all(
      providers.map((p) =>
        p.catalog[operation].pipe(
          Effect.tapError((cause) => Effect.logWarning(`Provider catalog ${operation} failed`).pipe(
            Effect.annotateLogs({ providerId: p.id, operation: `catalog-${operation}`, cause: String(cause).slice(0, 1_000) }),
          )),
          Effect.either,
        ),
      ),
    )
    return results.map((result, index): ProviderCatalogOutcome => result._tag === "Right"
      ? { _tag: "Success", providerId: providers[index]!.id, models: result.right }
      : { _tag: "Failure", providerId: providers[index]!.id, failure: result.left })
  })

export function buildFamilies(
  providerModels: readonly ProviderModel[],
  getModelFamily: (id: ModelFamilyId) => { readonly id: ModelFamilyId; readonly capabilities: { readonly vision: boolean } } | null,
): readonly { readonly id: ModelFamilyId; readonly capabilities: { readonly vision: boolean } }[] {
  const seen = new Set<ModelFamilyId>()
  const families: { readonly id: ModelFamilyId; readonly capabilities: { readonly vision: boolean } }[] = []
  for (const pm of providerModels) {
    if (!pm.modelFamilyId) continue
    if (!seen.has(pm.modelFamilyId)) {
      seen.add(pm.modelFamilyId)
      const family = getModelFamily(pm.modelFamilyId)
      if (family) families.push(family)
    }
  }
  return families
}
