import { Context, Effect, Layer, Ref } from "effect"
import type { ProviderModel } from "../lib/model/provider-model"
import { mergeProviderModels } from "./merge"
import { localDiscoveryCatalogueSource } from "./local-discovery/source"
import { modelsDevCatalogueSource } from "./models-dev/source"
import { openRouterCatalogueSource } from "./openrouter/source"
import { staticCatalogueSource } from "./static/source"
import type { CatalogueSource } from "./types"

export class ModelCatalogue extends Context.Tag("@magnitudedev/ai/ModelCatalogue")<
  ModelCatalogue,
  {
    readonly refresh: () => Effect.Effect<void>
    readonly getModels: (providerId: string) => Effect.Effect<readonly ProviderModel[]>
    readonly getAllModels: () => Effect.Effect<ReadonlyMap<string, readonly ProviderModel[]>>
  }
>() {}

function sourceFailedMessage(sourceId: string, error: unknown): string {
  const detail =
    error instanceof Error
      ? error.message
      : typeof error === "object" && error !== null && "_tag" in error
        ? String((error as { readonly _tag: unknown })._tag)
        : String(error)

  return `Catalogue source "${sourceId}" failed: ${detail}`
}

function mergeSourceResult(
  merged: Map<string, ProviderModel[]>,
  sourceModels: ReadonlyMap<string, readonly ProviderModel[]>,
): void {
  for (const [providerId, models] of sourceModels) {
    const existing = merged.get(providerId) ?? []
    merged.set(providerId, mergeProviderModels(existing, [...models]))
  }
}

export function makeModelCatalogueLive(
  sources: readonly CatalogueSource[],
): Layer.Layer<ModelCatalogue> {
  return Layer.effect(
    ModelCatalogue,
    Effect.gen(function* () {
      const state = yield* Ref.make<ReadonlyMap<string, readonly ProviderModel[]>>(new Map())

      const refresh = Effect.gen(function* () {
        const merged = new Map<string, ProviderModel[]>()

        for (const source of sources) {
          const result = yield* Effect.either(source.fetch())
          if (result._tag === "Right") {
            mergeSourceResult(merged, result.right)
          } else {
            yield* Effect.logDebug(sourceFailedMessage(source.id, result.left))
          }
        }

        yield* Ref.set(state, merged)
      })

      return {
        refresh: () => refresh,
        getModels: (providerId) =>
          Ref.get(state).pipe(Effect.map((modelsByProvider) => modelsByProvider.get(providerId) ?? [])),
        getAllModels: () => Ref.get(state),
      }
    }),
  )
}

const defaultSources: readonly CatalogueSource[] = [
  staticCatalogueSource,
  modelsDevCatalogueSource,
  openRouterCatalogueSource,
  localDiscoveryCatalogueSource,
]

export const ModelCatalogueLive = makeModelCatalogueLive(defaultSources)
