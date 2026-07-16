import { Context, Effect, Layer, Ref } from "effect"
import {
  ProviderClient,
  createProviderClient,
  type ProviderClientShape,
} from "@magnitudedev/sdk"
import {
  MagnitudeStorage,
  type AuthStorageShape,
  type MagnitudeStorageShape,
} from "@magnitudedev/storage"
import { LocalModelProviderSource } from "./local-inference/provider-source"

const resolveMagnitudeApiKey = (
  storage: MagnitudeStorageShape,
): Effect.Effect<string | null> => Effect.gen(function* () {
  const auth = yield* storage.auth.get("magnitude").pipe(Effect.orElseSucceed(() => null))
  if (auth?.type === "api" && auth.key.trim()) return auth.key
  const environmentKey = process.env.MAGNITUDE_API_KEY
  return environmentKey?.trim() ? environmentKey : null
})

export interface EndpointProviderAuthConfig {
  readonly endpoint: string
  readonly apiKey?: string
}

interface EndpointProviderAuthStorage {
  readonly auth: Pick<AuthStorageShape, "get">
}

/** Resolve endpoint auth without mutating storage during inspection. */
export const resolveEndpointProviderAuthFromStorage = (
  storage: EndpointProviderAuthStorage,
  providerId: string,
  defaultConfig?: EndpointProviderAuthConfig,
): Effect.Effect<EndpointProviderAuthConfig | null> => Effect.gen(function* () {
  const read = yield* storage.auth.get(providerId).pipe(Effect.either)
  if (read._tag === "Right" && read.right?.type === "endpoint") {
    const endpoint = read.right.endpoint.trim()
    if (endpoint) {
      return {
        endpoint,
        ...(read.right.apiKey ? { apiKey: read.right.apiKey } : {}),
      }
    }
  }
  return defaultConfig ?? null
})

export const resolveLlamaCppAuth = (storage: EndpointProviderAuthStorage) =>
  resolveEndpointProviderAuthFromStorage(storage, "llamacpp")

interface ProviderClientEntry {
  readonly sessionId: string | null
  readonly ref: Ref.Ref<ProviderClientShape>
  readonly client: ProviderClientShape
}

export const makeDelegatingProviderClient = (
  ref: Ref.Ref<ProviderClientShape>,
  runtimeConfig: ProviderClientShape["runtimeConfig"],
  sessionId: string | null,
): ProviderClientShape => ({
  catalog: {
    list: Ref.get(ref).pipe(Effect.flatMap((client) => client.catalog.list)),
    get: (providerId, providerModelId) => Ref.get(ref).pipe(
      Effect.flatMap((client) => client.catalog.get(providerId, providerModelId)),
    ),
    refresh: Ref.get(ref).pipe(Effect.flatMap((client) => client.catalog.refresh)),
  },
  catalogs: {
    list: Ref.get(ref).pipe(Effect.flatMap((client) => client.catalogs.list)),
    refresh: (providerId) => Ref.get(ref).pipe(Effect.flatMap((client) => client.catalogs.refresh(providerId))),
  },
  listProviders: Ref.get(ref).pipe(Effect.flatMap((client) => client.listProviders)),
  sessionId,
  resolveModel: (providerId, providerModelId, options) => Ref.get(ref).pipe(
    Effect.flatMap((client) => client.resolveModel(providerId, providerModelId, options)),
  ),
  webSearch: (query, schema) => Ref.get(ref).pipe(
    Effect.flatMap((client) => client.webSearch(query, schema)),
  ),
  usage: (query) => Ref.get(ref).pipe(Effect.flatMap((client) => client.usage(query))),
  runtimeConfig,
})

export interface ProviderClientRegistryApi {
  readonly shared: ProviderClientShape
  readonly session: (sessionId: string) => Effect.Effect<ProviderClientShape>
  readonly refreshAll: Effect.Effect<void>
  readonly remove: (sessionId: string) => Effect.Effect<void>
}

export class ProviderClientRegistry extends Context.Tag("ProviderClientRegistry")<
  ProviderClientRegistry,
  ProviderClientRegistryApi
>() {}

export const ProviderClientRegistryLive: Layer.Layer<
  ProviderClientRegistry,
  never,
  MagnitudeStorage | LocalModelProviderSource
> = Layer.effect(
  ProviderClientRegistry,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const llamacpp = yield* LocalModelProviderSource
    const entries = yield* Ref.make<ReadonlyMap<string, ProviderClientEntry>>(new Map())
    const lock = yield* Effect.makeSemaphore(1)

    const makeConcrete = (sessionId: string | null) => Effect.gen(function* () {
      const apiKey = yield* resolveMagnitudeApiKey(storage)
      const client = createProviderClient({
        ...(apiKey ? { apiKey } : {}),
        ...(sessionId ? { sessionId } : {}),
        llamacpp,
      })
      const withLiveLocalModels = (models: readonly import("@magnitudedev/ai").ProviderModel[]) =>
        llamacpp.catalog.list.pipe(
          Effect.map((local) => [...models.filter((model) => model.providerId !== "llamacpp"), ...local]),
        )
      const catalog: ProviderClientShape["catalog"] = {
        list: client.catalog.list.pipe(Effect.flatMap(withLiveLocalModels)),
        refresh: client.catalog.refresh.pipe(Effect.flatMap(withLiveLocalModels)),
        get: (providerId, providerModelId) => providerId === "llamacpp"
          ? llamacpp.catalog.get(providerId, providerModelId)
          : client.catalog.get(providerId, providerModelId),
      }
      const fileBacked: ProviderClientShape = { ...client, catalog }
      return fileBacked
    })

    const makeEntry = (sessionId: string | null) => Effect.gen(function* () {
      const concrete = yield* makeConcrete(sessionId)
      const ref = yield* Ref.make(concrete)
      return {
        sessionId,
        ref,
        client: makeDelegatingProviderClient(ref, concrete.runtimeConfig, sessionId),
      } satisfies ProviderClientEntry
    })

    const sharedEntry = yield* makeEntry(null)
    yield* Ref.set(entries, new Map([["shared", sharedEntry]]))

    return ProviderClientRegistry.of({
      shared: sharedEntry.client,
      session: (sessionId) => lock.withPermits(1)(Effect.gen(function* () {
        const key = `session:${sessionId}`
        const current = yield* Ref.get(entries)
        const existing = current.get(key)
        if (existing) return existing.client
        const entry = yield* makeEntry(sessionId)
        yield* Ref.set(entries, new Map(current).set(key, entry))
        return entry.client
      })),
      refreshAll: lock.withPermits(1)(Effect.gen(function* () {
        const current = yield* Ref.get(entries)
        yield* Effect.forEach(current.values(), (entry) => makeConcrete(entry.sessionId).pipe(
          Effect.flatMap((replacement) => Ref.set(entry.ref, replacement)),
        ), { concurrency: 4, discard: true })
      })),
      remove: (sessionId) => lock.withPermits(1)(Ref.update(entries, (current) => {
        const next = new Map(current)
        next.delete(`session:${sessionId}`)
        return next
      })),
    })
  }),
)

export const SharedProviderClientLive: Layer.Layer<ProviderClient, never, ProviderClientRegistry> =
  Layer.effect(ProviderClient, Effect.map(ProviderClientRegistry, (registry) => registry.shared))
