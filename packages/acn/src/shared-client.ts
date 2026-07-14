import { Context, Effect, Layer, Ref } from "effect"
import {
  ProviderClient,
  DEFAULT_LLAMACPP_ENDPOINT,
  SUPPORTED_PROVIDER_DEFINITIONS,
  createProviderClient,
  type ProviderConnectionConfig,
  type ProviderClientShape,
} from "@magnitudedev/sdk"
import { makeFileBackedModelCatalog } from "@magnitudedev/ai"
import { GlobalStorage, MagnitudeStorage, type MagnitudeStorageShape } from "@magnitudedev/storage"
import { FetchHttpClient } from "@effect/platform"
import type { ProviderAuth, ProviderAuthSummary } from "@magnitudedev/protocol"

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

function maskApiKey(key: string): string {
  const trimmed = key.trim()
  if (trimmed.length <= 8) return "*".repeat(Math.max(trimmed.length, 4))
  return `${trimmed.slice(0, 4)}...${trimmed.slice(-4)}`
}

function storedApiKey(
  auths: Readonly<Record<string, ProviderAuth>>,
  providerId: string,
): string | null {
  const direct = auths[providerId]
  if (direct?.type === "api" && direct.key.trim()) return direct.key.trim()
  return null
}

export interface ResolvedProviderConfiguration {
  readonly magnitudeApiKey: string | null
  readonly connections: Readonly<Record<string, ProviderConnectionConfig>>
  readonly authSummaries: readonly ProviderAuthSummary[]
}

/** Resolve every provider from environment, auth.json, and local defaults. */
export const resolveProviderConfiguration = (
  storage: MagnitudeStorageShape,
): Effect.Effect<ResolvedProviderConfiguration, never, never> =>
  Effect.gen(function* () {
    const auths = yield* storage.auth.loadAll().pipe(
      Effect.catchAll(() => Effect.succeed({} as Record<string, ProviderAuth>)),
    )
    const existingLlama = auths.llamacpp
    const llama = yield* resolveLlamaCppAuth(storage)
    const connections: Record<string, ProviderConnectionConfig> = {}
    const summaries: ProviderAuthSummary[] = []
    const useLocal = process.env.MAGNITUDE_USE_LOCAL?.trim().toLowerCase()
    const localMagnitudeKey = ["1", "true", "yes", "on"].includes(useLocal ?? "")
      ? process.env.MAGNITUDE_LOCAL_API_KEY?.trim() || undefined
      : undefined

    for (const definition of SUPPORTED_PROVIDER_DEFINITIONS) {
      if (definition.id === "llamacpp") {
        if (llama) {
          const source = existingLlama?.type === "endpoint"
            && (
              existingLlama.endpoint.trim() !== DEFAULT_LLAMACPP_ENDPOINT
              || Boolean(existingLlama.apiKey?.trim())
            )
            ? "file" as const
            : "default" as const
          connections.llamacpp = {
            endpoint: llama.endpoint,
            ...(llama.apiKey ? { apiKey: llama.apiKey } : {}),
            authSource: source,
          }
          summaries.push({
            providerId: definition.id,
            type: "endpoint",
            configured: true,
            source,
            endpoint: llama.endpoint,
            ...(llama.apiKey ? { maskedKey: maskApiKey(llama.apiKey) } : {}),
          })
        }
        continue
      }

      const configuredEnvKey = definition.environmentKeys
        .map((name) => process.env[name]?.trim())
        .find((value): value is string => Boolean(value))
      const envKey = definition.id === "magnitude"
        ? localMagnitudeKey ?? configuredEnvKey
        : configuredEnvKey
      const fileKey = storedApiKey(auths, definition.id)
      const key = envKey ?? fileKey
      const source = envKey ? "env" as const : fileKey ? "file" as const : "none" as const

      if (key) {
        connections[definition.id] = { apiKey: key, authSource: source }
      }
      summaries.push({
        providerId: definition.id,
        type: definition.authKind,
        configured: Boolean(key),
        source,
        ...(key ? { maskedKey: maskApiKey(key) } : {}),
      })
    }

    const magnitudeApiKey = connections.magnitude?.apiKey ?? null

    return { magnitudeApiKey, connections, authSummaries: summaries }
  })

/**
 * Build a fresh file-backed `ProviderClientShape` using the resolved provider
 * auth and the global model cache path. The catalog is intentionally refreshed
 * so the file cache is populated with every currently available provider.
 */
export const makeSharedProviderClient = (
  resolved: ResolvedProviderConfiguration,
): Effect.Effect<ProviderClientShape, never, GlobalStorage> =>
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage
    const client = createProviderClient({
      ...(resolved.magnitudeApiKey ? { apiKey: resolved.magnitudeApiKey } : {}),
      providerConnections: resolved.connections,
    })
    const preserveProviderIds = [
      ...(resolved.magnitudeApiKey ? ["magnitude"] : []),
      ...Object.keys(resolved.connections).filter((providerId) => providerId !== "llamacpp"),
    ]
    const fileCatalog = makeFileBackedModelCatalog(
      client.catalog,
      globalStorage.paths.modelCacheFile,
      undefined,
      undefined,
      { preserveProviderIds },
    )
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
    const resolved = yield* resolveProviderConfiguration(storage)
    const client = yield* makeSharedProviderClient(resolved)
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
