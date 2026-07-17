import { Context, Effect, Layer } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import type {
  Provider,
  ModelCatalog,
  ProviderModel,
  BoundModel,
  BaseCallOptions,
  ProviderModelBindOptions,
  ProviderId,
  ProviderModelId,
  ModelPropertyDiscoveryRequest,
  ModelDiscoveryOperationId,
  ModelPropertyDiscoveryError,
} from "@magnitudedev/ai"
import type { MagnitudeProviderInstance } from "./magnitude/provider"
import { inspectProviderCatalogs, makeAggregatedCatalog, type ProviderCatalogOutcome } from "./catalog-aggregator"

export type AuthStatus =
  | { readonly _tag: "authenticated" }
  | { readonly _tag: "no_auth_required" }
  | { readonly _tag: "not_configured"; readonly reason: string }

export interface ProviderInfo {
  readonly id: ProviderId
  readonly displayName: string
  readonly authStatus: AuthStatus
  readonly status?: "ok" | "loading" | "not_found" | "error"
  readonly message?: string
  readonly hint?: string
}

export interface DiscoverableProviderInstance {
  readonly provider: Pick<Provider, "id" | "displayName" | "bindModel" | "catalog" | "discoverModelProperties">
  readonly authStatus?: AuthStatus
  readonly checkStatus: Effect.Effect<{
    readonly status: "ok" | "loading" | "not_found" | "error"
    readonly message?: string
    readonly hint?: string
  }, never, HttpClient.HttpClient>
}

export interface ProviderRegistryService {
  readonly listProviderIds: Effect.Effect<readonly ProviderId[]>
  readonly listProviders: Effect.Effect<readonly ProviderInfo[], never, HttpClient.HttpClient>
  readonly aggregatedCatalog: ModelCatalog<ProviderModel>
  readonly catalogs: {
    readonly list: Effect.Effect<readonly ProviderCatalogOutcome[], never, HttpClient.HttpClient>
    readonly refresh: (providerId?: ProviderId) => Effect.Effect<readonly ProviderCatalogOutcome[], never, HttpClient.HttpClient>
  }
  /**
   * Resolve a model from any registered provider.
   * Dispatches to the correct provider's `model()` method.
   */
  readonly resolveModel: (
    providerId: ProviderId,
    providerModelId: ProviderModelId,
    options?: ProviderModelBindOptions,
  ) => Effect.Effect<BoundModel<BaseCallOptions>, never, never>
  readonly discoverModelProperties: (
    providerId: ProviderId,
    request: ModelPropertyDiscoveryRequest,
  ) => Effect.Effect<ModelDiscoveryOperationId, ModelPropertyDiscoveryError>
}

export class ProviderRegistry extends Context.Tag("ProviderRegistry")<
  ProviderRegistry,
  ProviderRegistryService
>() {}

/**
 * Create a registry from configured provider instances.
 * Only non-null instances are activated.
 */
export function makeProviderRegistry(config: {
  readonly magnitude: MagnitudeProviderInstance | null
  readonly discoverableProviders?: readonly DiscoverableProviderInstance[]
}): ProviderRegistryService {
  const providers = new Map<ProviderId, Pick<Provider, "id" | "bindModel" | "catalog" | "discoverModelProperties">>()
  const providerInfos: ProviderInfo[] = []

  if (config.magnitude) {
    providers.set(config.magnitude.provider.id, config.magnitude.provider)
    providerInfos.push({
      id: config.magnitude.provider.id,
      displayName: "Magnitude",
      authStatus: config.magnitude.authentication._tag === "Configured"
        ? { _tag: "authenticated" }
        : { _tag: "not_configured", reason: "Magnitude authentication is not configured" },
    })
  }

  for (const instance of config.discoverableProviders ?? []) {
    providers.set(instance.provider.id, instance.provider)
  }

  const activeProviders = [...providers.values()].map((p) => ({ id: p.id, catalog: p.catalog }))
  const aggregatedCatalog = makeAggregatedCatalog(activeProviders)

  return {
    listProviderIds: Effect.succeed([...providers.keys()]),
    listProviders: Effect.gen(function* () {
      const infos = [...providerInfos]
      for (const instance of config.discoverableProviders ?? []) {
        const result = yield* instance.checkStatus
        infos.push({
          id: instance.provider.id,
          displayName: instance.provider.displayName,
          authStatus: instance.authStatus ?? { _tag: "no_auth_required" },
          status: result.status,
          ...(result.message ? { message: result.message } : {}),
          ...(result.hint ? { hint: result.hint } : {}),
        })
      }
      return infos
    }),
    aggregatedCatalog,
    catalogs: {
      list: inspectProviderCatalogs(activeProviders, "list"),
      refresh: (providerId) => inspectProviderCatalogs(
        providerId === undefined ? activeProviders : activeProviders.filter((provider) => provider.id === providerId),
        "refresh",
      ),
    },
    resolveModel: (providerId, providerModelId, options) =>
      Effect.gen(function* () {
        const provider = providers.get(providerId)
        if (!provider) return yield* Effect.die(`Unknown provider: ${providerId}`)
        return yield* provider.bindModel(providerModelId, options)
      }),
    discoverModelProperties: (providerId, request) => Effect.gen(function* () {
      const provider = providers.get(providerId)
      if (!provider) return yield* Effect.die(`Unknown provider: ${providerId}`)
      return yield* provider.discoverModelProperties(request)
    }),
  }
}

export const ProviderRegistryLive = (config: {
  readonly magnitude: MagnitudeProviderInstance | null
  readonly discoverableProviders?: readonly DiscoverableProviderInstance[]
}) => Layer.succeed(ProviderRegistry, makeProviderRegistry(config))
