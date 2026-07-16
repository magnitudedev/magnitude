import { Effect } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import {
  ModelCatalogError,
  type ModelCatalog,
  type ProviderModel,
} from "@magnitudedev/ai"

/**
 * Merge multiple provider catalogs into a single aggregated catalog.
 * The aggregated catalog type-erases to ProviderModel (base type).
 */
export function makeAggregatedCatalog(
  providers: readonly { readonly id: string; readonly catalog: ModelCatalog<any> }[],
): ModelCatalog<ProviderModel> {
  const list: ModelCatalog<ProviderModel>["list"] = Effect.gen(function* () {
    const results = yield* Effect.all(
      providers.map((p) =>
        p.catalog.list.pipe(Effect.catchAll(() => Effect.succeed([] as readonly any[]))),
      ),
    )
    return results.flat() as readonly ProviderModel[]
  })

  const get: ModelCatalog<ProviderModel>["get"] = (providerId, providerModelId) =>
    Effect.gen(function* () {
      const provider = providers.find((p) => p.id === providerId)
      if (!provider) return yield* new ModelCatalogError({ message: `Unknown provider: ${providerId}` })
      return yield* provider.catalog.get(providerId, providerModelId) as Effect.Effect<ProviderModel, ModelCatalogError, HttpClient.HttpClient>
    })

  const refresh: ModelCatalog<ProviderModel>["refresh"] = Effect.gen(function* () {
    const results = yield* Effect.all(
      providers.map((p) =>
        p.catalog.refresh.pipe(Effect.catchAll(() => Effect.succeed([] as readonly any[]))),
      ),
    )
    return results.flat() as readonly ProviderModel[]
  })

  return { list, get, refresh }
}

export function buildFamilies(
  providerModels: readonly ProviderModel[],
  getModelFamily: (id: string) => { readonly id: string; readonly capabilities: { readonly vision: boolean } } | null,
): readonly { readonly id: string; readonly capabilities: { readonly vision: boolean } }[] {
  const seen = new Set<string>()
  const families: { readonly id: string; readonly capabilities: { readonly vision: boolean } }[] = []
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
