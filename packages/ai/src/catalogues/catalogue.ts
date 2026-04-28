import { Context, Effect, Layer, Option, Ref } from "effect"
import type { ProviderModel } from "../lib/model/provider-model"
import { getAllProviders } from "../providers/registry"
import { mergeProviderModels } from "./merge"
import { modelsDevCatalogueSource } from "./models-dev/source"
import { makeLocalDiscoverySource } from "./local-discovery/source"
import { openRouterCatalogueSource } from "./openrouter/source"
import { staticCatalogueSource } from "./static/source"

export class ModelCatalogue extends Context.Tag("@magnitudedev/ai/ModelCatalogue")<
  ModelCatalogue,
  {
    readonly refresh: () => Effect.Effect<void>
    readonly getModels: (providerId: string) => Effect.Effect<readonly ProviderModel[]>
    readonly getAllModels: () => Effect.Effect<ReadonlyMap<string, readonly ProviderModel[]>>
  }
>() {}

function sourceFailedMessage(sourceId: string, providerId: string, error: unknown): string {
  const detail =
    error instanceof Error
      ? error.message
      : typeof error === "object" && error !== null && "_tag" in error
        ? String((error as { readonly _tag: unknown })._tag)
        : String(error)

  return `Catalogue source "${sourceId}" failed for provider "${providerId}": ${detail}`
}

export const ModelCatalogueLive = Layer.effect(
  ModelCatalogue,
  Effect.gen(function* () {
    const state = yield* Ref.make<ReadonlyMap<string, readonly ProviderModel[]>>(
      new Map(
        getAllProviders().map((provider) => [provider.id, [...provider.models]] as const),
      ),
    )

    const refreshProvider = (providerId: string): Effect.Effect<readonly ProviderModel[]> =>
      Effect.gen(function* () {
        const provider = getAllProviders().find((entry) => entry.id === providerId)
        if (!provider) {
          return []
        }

        const successfulLayers: ProviderModel[][] = []

        const staticResult = yield* Effect.either(staticCatalogueSource.fetch)
        if (staticResult._tag === "Right") {
          successfulLayers.push(
            staticResult.right.filter((model) => model.providerId === provider.id).map((model) => ({ ...model })),
          )
        }

        const dynamicSources = [
          provider.id === "openrouter" ? openRouterCatalogueSource : null,
          provider.family === "cloud" && provider.id !== "magnitude" ? modelsDevCatalogueSource : null,
          makeLocalDiscoverySource(provider),
        ].filter((source): source is NonNullable<typeof source> => source !== null)

        for (const source of dynamicSources) {
          const result = yield* Effect.either(source.fetch)
          if (result._tag === "Right") {
            successfulLayers.push(result.right.map((model) => ({ ...model })))
          } else {
            yield* Effect.logDebug(sourceFailedMessage(source.id, provider.id, result.left))
          }
        }

        if (successfulLayers.length === 0) {
          return [...provider.models]
        }

        return mergeProviderModels(successfulLayers[0] ?? [], ...successfulLayers.slice(1))
      })

    const refresh = Effect.gen(function* () {
      const next = new Map<string, readonly ProviderModel[]>()

      for (const provider of getAllProviders()) {
        const models = yield* refreshProvider(provider.id)
        next.set(provider.id, models)
      }

      yield* Ref.set(state, next)
    })

    const getModels = (providerId: string): Effect.Effect<readonly ProviderModel[]> =>
      Ref.get(state).pipe(
        Effect.map((modelsByProvider) => modelsByProvider.get(providerId) ?? []),
      )

    const getAllModels = (): Effect.Effect<ReadonlyMap<string, readonly ProviderModel[]>> =>
      Ref.get(state)

    return {
      refresh: () => refresh,
      getModels,
      getAllModels,
    }
  }),
)
