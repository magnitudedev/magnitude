import { Context, Effect, Layer } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import type {
  Provider,
  ModelCatalog,
  ProviderModel,
  BoundModel,
  BaseCallOptions,
  ProviderModelBindOptions,
} from "@magnitudedev/ai"
import type { MagnitudeProviderInstance } from "./magnitude/provider"
import { makeAggregatedCatalog } from "./catalog-aggregator"

export type AuthStatus =
  | { readonly _tag: "authenticated" }
  | { readonly _tag: "no_auth_required" }
  | { readonly _tag: "not_configured"; readonly reason: string }

export interface ProviderInfo {
  readonly id: string
  readonly displayName: string
  readonly authStatus: AuthStatus
  readonly status?: "ok" | "loading" | "not_found" | "error"
  readonly message?: string
  readonly hint?: string
}

export interface DiscoverableProviderInstance {
  readonly provider: Pick<Provider, "id" | "displayName" | "bindModel" | "catalog">
  readonly authStatus?: AuthStatus
  readonly checkStatus: Effect.Effect<{
    readonly status: "ok" | "loading" | "not_found" | "error"
    readonly message?: string
    readonly hint?: string
  }, never, HttpClient.HttpClient>
}

export interface ProviderRegistryService {
  readonly listProviderIds: Effect.Effect<readonly string[]>
  readonly listProviders: Effect.Effect<readonly ProviderInfo[], never, HttpClient.HttpClient>
  readonly aggregatedCatalog: ModelCatalog<ProviderModel>
  /**
   * Resolve a model from any registered provider.
   * Dispatches to the correct provider's `model()` method.
   */
  readonly resolveModel: (
    providerId: string,
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ) => Effect.Effect<BoundModel<BaseCallOptions>, never, never>
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
  const providers = new Map<string, Pick<Provider, "id" | "bindModel" | "catalog">>()
  const providerInfos: ProviderInfo[] = []

  if (config.magnitude) {
    providers.set("magnitude", config.magnitude.provider)
    providerInfos.push({ id: "magnitude", displayName: "Magnitude", authStatus: { _tag: "authenticated" } })
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
    resolveModel: (providerId, providerModelId, options) =>
      Effect.gen(function* () {
        const provider = providers.get(providerId)
        if (!provider) return yield* Effect.die(`Unknown provider: ${providerId}`)
        return yield* provider.bindModel(providerModelId, options)
      }),
  }
}

export const ProviderRegistryLive = (config: {
  readonly magnitude: MagnitudeProviderInstance | null
  readonly discoverableProviders?: readonly DiscoverableProviderInstance[]
}) => Layer.succeed(ProviderRegistry, makeProviderRegistry(config))
