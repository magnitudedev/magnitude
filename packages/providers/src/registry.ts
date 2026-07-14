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

export type AuthSource = "env" | "file" | "default" | "none"
export type ProviderAuthKind = "api" | "endpoint" | "none"

export interface ProviderInfo {
  readonly id: string
  readonly displayName: string
  readonly authStatus: AuthStatus
  readonly authKind?: ProviderAuthKind
  readonly authSource?: AuthSource
  readonly status?: "ok" | "loading" | "not_found" | "error"
  readonly message?: string
  readonly hint?: string
}

export interface ConfiguredProviderInstance {
  readonly provider: Pick<Provider, "id" | "displayName" | "bindModel" | "catalog">
  readonly authStatus?: AuthStatus
  readonly authKind?: ProviderAuthKind
  readonly authSource?: AuthSource
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
  readonly configuredProviders?: readonly ConfiguredProviderInstance[]
  readonly discoverableProviders?: readonly DiscoverableProviderInstance[]
  readonly providerDefinitions?: readonly ProviderInfo[]
}): ProviderRegistryService {
  const providers = new Map<string, Pick<Provider, "id" | "bindModel" | "catalog">>()
  const providerInfos = new Map<string, ProviderInfo>()

  for (const definition of config.providerDefinitions ?? []) {
    providerInfos.set(definition.id, definition)
  }

  if (config.magnitude) {
    const existing = providerInfos.get("magnitude")
    providers.set("magnitude", config.magnitude.provider)
    providerInfos.set("magnitude", {
      id: "magnitude",
      displayName: "Magnitude",
      authStatus: { _tag: "authenticated" },
      authKind: "api",
      ...(existing?.authSource ? { authSource: existing.authSource } : {}),
    })
  }

  for (const instance of config.configuredProviders ?? []) {
    providers.set(instance.provider.id, instance.provider)
    providerInfos.set(instance.provider.id, {
      id: instance.provider.id,
      displayName: instance.provider.displayName,
      authStatus: instance.authStatus ?? { _tag: "authenticated" },
      ...(instance.authKind ? { authKind: instance.authKind } : {}),
      ...(instance.authSource ? { authSource: instance.authSource } : {}),
    })
  }

  for (const instance of config.discoverableProviders ?? []) {
    providers.set(instance.provider.id, instance.provider)
    const existing = providerInfos.get(instance.provider.id)
    providerInfos.set(instance.provider.id, {
      id: instance.provider.id,
      displayName: instance.provider.displayName,
      authStatus: instance.authStatus ?? existing?.authStatus ?? { _tag: "no_auth_required" },
      authKind: existing?.authKind ?? "endpoint",
      ...(existing?.authSource ? { authSource: existing.authSource } : {}),
    })
  }

  const activeProviders = [...providers.values()].map((p) => ({ id: p.id, catalog: p.catalog }))
  const aggregatedCatalog = makeAggregatedCatalog(activeProviders)

  return {
    listProviderIds: Effect.succeed([...providers.keys()]),
    listProviders: Effect.gen(function* () {
      const infos = new Map(providerInfos)
      for (const instance of config.configuredProviders ?? []) {
        const result = yield* Effect.either(instance.provider.catalog.list)
        const existing = infos.get(instance.provider.id)
        infos.set(instance.provider.id, {
          id: instance.provider.id,
          displayName: instance.provider.displayName,
          authStatus: instance.authStatus ?? existing?.authStatus ?? { _tag: "authenticated" },
          ...(existing?.authKind ? { authKind: existing.authKind } : {}),
          ...(existing?.authSource ? { authSource: existing.authSource } : {}),
          status: result._tag === "Right" ? "ok" : "error",
          ...(result._tag === "Left" ? { message: result.left.message } : {}),
        })
      }
      for (const instance of config.discoverableProviders ?? []) {
        const result = yield* instance.checkStatus
        const existing = infos.get(instance.provider.id)
        infos.set(instance.provider.id, {
          id: instance.provider.id,
          displayName: instance.provider.displayName,
          authStatus: instance.authStatus ?? existing?.authStatus ?? { _tag: "no_auth_required" },
          ...(existing?.authKind ? { authKind: existing.authKind } : {}),
          ...(existing?.authSource ? { authSource: existing.authSource } : {}),
          status: result.status,
          ...(result.message ? { message: result.message } : {}),
          ...(result.hint ? { hint: result.hint } : {}),
        })
      }
      return [...infos.values()]
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
  readonly configuredProviders?: readonly ConfiguredProviderInstance[]
  readonly discoverableProviders?: readonly DiscoverableProviderInstance[]
  readonly providerDefinitions?: readonly ProviderInfo[]
}) => Layer.succeed(ProviderRegistry, makeProviderRegistry(config))
