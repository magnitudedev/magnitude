import { Context, Effect, Layer, Ref } from "effect"
import {
  ProviderClient,
  createProviderClient,
  type ProviderClientShape,
} from "@magnitudedev/sdk"
import { makeFileBackedModelCatalog } from "@magnitudedev/ai"
import { GlobalStorage, MagnitudeStorage, type MagnitudeStorageShape } from "@magnitudedev/storage"
import { FetchHttpClient } from "@effect/platform"

/**
 * Resolve the API key from environment + storage. Returns `null` when no key is
 * configured.
 */
const resolveApiKeyFromStorage = (
  storage: MagnitudeStorageShape,
): Effect.Effect<string | null, never, never> =>
  Effect.gen(function* () {
    const auth = yield* storage.auth.get("magnitude").pipe(
      Effect.catchAll(() => Effect.succeed(null)),
    )
    if (auth?.type === "api" && auth.key.trim()) return auth.key

    const envKey = process.env.MAGNITUDE_API_KEY
    if (envKey?.trim()) return envKey

    return null
  })

export interface LlamaCppAuthConfig {
  readonly endpoint: string
  readonly apiKey?: string
}

export const resolveLlamaCppAuthFromStorage = (
  storage: MagnitudeStorageShape,
): Effect.Effect<LlamaCppAuthConfig | null, never, never> =>
  Effect.gen(function* () {
    const auth = yield* storage.auth.get("llamacpp").pipe(
      Effect.catchAll(() => Effect.succeed(null)),
    )
    if (auth?.type !== "endpoint") return null

    const endpoint = auth.endpoint.trim()
    if (!endpoint) return null
    return {
      endpoint,
      ...(auth.apiKey ? { apiKey: auth.apiKey } : {}),
    }
  })

/**
 * Build a fresh file-backed `ProviderClientShape` using the resolved provider
 * auth and the global model cache path. The catalog is intentionally refreshed
 * so the file cache is populated with every currently available provider.
 */
export const makeClientWithAuth = (
  apiKey: string | null,
  llamacpp: LlamaCppAuthConfig | null,
): Effect.Effect<ProviderClientShape, never, GlobalStorage> =>
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage
    const client = createProviderClient({
      ...(apiKey ? { apiKey } : {}),
      ...(llamacpp
        ? {
            llamacppEndpoint: llamacpp.endpoint,
            ...(llamacpp.apiKey ? { llamacppApiKey: llamacpp.apiKey } : {}),
          }
        : {}),
    })
    const fileCatalog = makeFileBackedModelCatalog(client.catalog, globalStorage.paths.modelCacheFile)
    const fileBackedClient: ProviderClientShape = { ...client, catalog: fileCatalog }
    yield* fileBackedClient.catalog.refresh.pipe(
      Effect.provide(FetchHttpClient.layer),
      Effect.ignore,
    )
    return fileBackedClient
  })

/**
 * Mutable reference to the current shared ACN `ProviderClientShape`. This
 * lets the client be recreated (with a fresh, key-appropriate model cache) at
 * runtime — specifically after `UpdateProviderAuth` — while the `ProviderClient`
 * service remains in context.
 */
export class SharedMagnitudeClientRef extends Context.Tag("SharedMagnitudeClientRef")<
  SharedMagnitudeClientRef,
  Ref.Ref<ProviderClientShape>
>() {}

export const SharedMagnitudeClientRefLive = Layer.effect(
  SharedMagnitudeClientRef,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const apiKey = yield* resolveApiKeyFromStorage(storage)
    const llamacpp = yield* resolveLlamaCppAuthFromStorage(storage)
    const client = yield* makeClientWithAuth(apiKey, llamacpp)
    return yield* Ref.make<ProviderClientShape>(client)
  }),
)

/**
 * Build a wrapper client that delegates all calls to the current value of the
 * shared client ref. This means consumers that `yield* ProviderClient` always
 * see the latest client after `UpdateProviderAuth` recreates it.
 */
const wrapClientRef = (
  ref: Ref.Ref<ProviderClientShape>,
  runtimeConfig: ProviderClientShape["runtimeConfig"],
): ProviderClientShape => {
  return {
    catalog: {
      list: Effect.gen(function* () {
        const client = yield* ref.get
        return yield* client.catalog.list
      }),
      get: (providerId, providerModelId) => Effect.gen(function* () {
        const client = yield* ref.get
        return yield* client.catalog.get(providerId, providerModelId)
      }),
      refresh: Effect.gen(function* () {
        const client = yield* ref.get
        return yield* client.catalog.refresh
      }),
    },
    listProviders: Effect.gen(function* () {
      const client = yield* ref.get
      return yield* client.listProviders
    }),
    sessionId: null,
    resolveModel: (providerId, providerModelId, options) =>
      Effect.gen(function* () {
        const client = yield* ref.get
        return yield* client.resolveModel(providerId, providerModelId, options)
      }),
    webSearch: (query, schema) =>
      Effect.gen(function* () {
        const client = yield* ref.get
        return yield* client.webSearch(query, schema)
      }),
    balance: (query) =>
      Effect.gen(function* () {
        const client = yield* ref.get
        return yield* client.balance(query)
      }),
    runtimeConfig,
  }
}

/**
 * A single shared `ProviderClient` at the ACN daemon level. The underlying
 * client is held in `SharedMagnitudeClientRef` and can be recreated on auth
 * change so subsequent calls use the new auth and refreshed model cache.
 */
export const SharedMagnitudeClientLive = Layer.effect(
  ProviderClient,
  Effect.gen(function* () {
    const ref = yield* SharedMagnitudeClientRef
    const initial = yield* ref.get
    return wrapClientRef(ref, initial.runtimeConfig)
  }),
)
