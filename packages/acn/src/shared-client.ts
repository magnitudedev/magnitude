import { Context, Effect, Layer, Ref } from "effect"
import {
  ProviderClient,
  DEFAULT_LLAMACPP_ENDPOINT,
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
const resolveMagnitudeApiKeyFromStorage = (
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

export interface EndpointProviderAuthConfig {
  readonly endpoint: string
  readonly apiKey?: string
}

/** Resolve endpoint auth and seed a missing entry with the provider default. */
export const resolveEndpointProviderAuthFromStorage = (
  storage: MagnitudeStorageShape,
  providerId: string,
  defaultConfig?: EndpointProviderAuthConfig,
): Effect.Effect<EndpointProviderAuthConfig | null, never, never> =>
  Effect.gen(function* () {
    const read = yield* storage.auth.get(providerId).pipe(
      Effect.map((auth) => ({ _tag: "success" as const, auth })),
      Effect.catchAll(() => Effect.succeed({ _tag: "error" as const })),
    )
    if (read._tag === "success" && read.auth?.type === "endpoint") {
      const endpoint = read.auth.endpoint.trim()
      if (endpoint) {
        return {
          endpoint,
          ...(read.auth.apiKey ? { apiKey: read.auth.apiKey } : {}),
        }
      }
    }

    if (read._tag === "success" && read.auth === undefined && defaultConfig) {
      yield* storage.auth.set(providerId, {
        type: "endpoint",
        endpoint: defaultConfig.endpoint,
        ...(defaultConfig.apiKey ? { apiKey: defaultConfig.apiKey } : {}),
      }).pipe(
        Effect.catchAll(() => Effect.logWarning("Failed to persist default provider endpoint").pipe(
          Effect.annotateLogs({ providerId }),
        )),
      )
    }

    return defaultConfig ?? null
  })

export const resolveLlamaCppAuth = (storage: MagnitudeStorageShape) =>
  resolveEndpointProviderAuthFromStorage(storage, "llamacpp", {
    endpoint: DEFAULT_LLAMACPP_ENDPOINT,
  })

/**
 * Build a fresh file-backed `ProviderClientShape` using the resolved provider
 * auth and the global model cache path. The catalog is intentionally refreshed
 * so the file cache is populated with every currently available provider.
 */
export const makeSharedProviderClient = (
  magnitudeApiKey: string | null,
  llamacpp: EndpointProviderAuthConfig | null,
): Effect.Effect<ProviderClientShape, never, GlobalStorage> =>
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage
    const client = createProviderClient({
      ...(magnitudeApiKey ? { apiKey: magnitudeApiKey } : {}),
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
export class SharedProviderClientRef extends Context.Tag("SharedProviderClientRef")<
  SharedProviderClientRef,
  Ref.Ref<ProviderClientShape>
>() {}

export const SharedProviderClientRefLive = Layer.effect(
  SharedProviderClientRef,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const apiKey = yield* resolveMagnitudeApiKeyFromStorage(storage)
    const llamacpp = yield* resolveLlamaCppAuth(storage)
    const client = yield* makeSharedProviderClient(apiKey, llamacpp)
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
 * client is held in `SharedProviderClientRef` and can be recreated on auth
 * change so subsequent calls use the new auth and refreshed model cache.
 */
export const SharedProviderClientLive = Layer.effect(
  ProviderClient,
  Effect.gen(function* () {
    const ref = yield* SharedProviderClientRef
    const initial = yield* ref.get
    return wrapClientRef(ref, initial.runtimeConfig)
  }),
)
