import { Context, Effect, Layer, Ref, Schema, Stream, SubscriptionRef } from "effect"
import type { ToolAvailabilityState } from "@magnitudedev/agent"
import {
  ProviderClient,
  createCustomEndpointProvider,
  createProviderClient,
  type ProviderClientShape,
  type WebSearchSource,
} from "@magnitudedev/providers/client"
import {
  MagnitudeStorage,
  CustomEndpointNameSchema,
  type AuthStorageShape,
  type MagnitudeStorageShape,
} from "@magnitudedev/storage"
import { IcnProvider, createLocalProvider } from "@magnitudedev/icn/provider"
import { ModelSelection } from "./model-selection"
import { SlotIdSchema } from "@magnitudedev/acn-protocol"
import { CustomEndpoints } from "./custom-endpoints"

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
  discoverModelProperties: (providerId, request) => Ref.get(ref).pipe(
    Effect.flatMap((client) => client.discoverModelProperties(providerId, request)),
  ),
  requestAttribution: (providerId, providerModelId, key) => ({
    key,
    requestStarted: Ref.get(ref).pipe(
      Effect.flatMap((client) => client.requestAttribution(providerId, providerModelId, key).requestStarted),
    ),
  }),
  webSearchSource: Ref.get(ref).pipe(
    Effect.flatMap((client) => client.webSearchSource),
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
  readonly toolAvailability: Effect.Effect<ToolAvailabilityState>
  readonly toolAvailabilityChanges: Stream.Stream<ToolAvailabilityState>
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
  MagnitudeStorage | IcnProvider | ModelSelection | CustomEndpoints
> = Layer.effect(
  ProviderClientRegistry,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const local = createLocalProvider(yield* IcnProvider)
    const modelSelection = yield* ModelSelection
    const customEndpoints = yield* CustomEndpoints
    const entries = yield* Ref.make<ReadonlyMap<string, ProviderClientEntry>>(new Map())
    const lock = yield* Effect.makeSemaphore(1)

    const makeConcrete = (sessionId: string | null) => Effect.gen(function* () {
      const apiKey = yield* resolveMagnitudeApiKey(storage)
      const declarations = yield* customEndpoints.get
      const customProviders = Object.entries(declarations).map(([name, declaration]) =>
        createCustomEndpointProvider(CustomEndpointNameSchema.make(name), declaration))
      const client = createProviderClient({
        ...(apiKey ? { apiKey } : {}),
        ...(sessionId ? { sessionId } : {}),
        discoverableProviders: [{ ...local, kind: "Local" }, ...customProviders],
      })
      return {
        ...client,
        requestAttribution: (providerId, providerModelId, key) => {
          const attribution = client.requestAttribution(providerId, providerModelId, key)
          return {
            key,
            requestStarted: attribution.requestStarted.pipe(Effect.zipRight(
              Schema.is(SlotIdSchema)(key)
                ? modelSelection.recordUse(key, { providerId, providerModelId }).pipe(Effect.ignore)
                : Effect.void,
            )),
          }
        },
      } satisfies ProviderClientShape
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
    const initialSource = yield* sharedEntry.client.webSearchSource
    const toolAvailability = yield* SubscriptionRef.make<ToolAvailabilityState>(
      toolAvailabilityFromSource(initialSource),
    )
    const publishToolAvailability = (next: ToolAvailabilityState) =>
      SubscriptionRef.set(toolAvailability, next)

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
      toolAvailability: SubscriptionRef.get(toolAvailability),
      toolAvailabilityChanges: toolAvailability.changes,
      refreshAll: lock.withPermits(1)(Effect.gen(function* () {
        const current = yield* Ref.get(entries)
        const replacements = yield* Effect.forEach(
          current.values(),
          (entry) => makeConcrete(entry.sessionId).pipe(
            Effect.map((client) => ({ entry, client })),
          ),
          { concurrency: 4 },
        )
        const nextSource = replacements.length > 0
          ? yield* replacements[0]!.client.webSearchSource
          : "unavailable" as const
        const previous = yield* SubscriptionRef.get(toolAvailability)
        const next = nextToolAvailability(previous, nextSource)
        const sourceChanged = next !== previous

        if (sourceChanged && nextSource === "unavailable") {
          yield* publishToolAvailability(next)
        }
        yield* Effect.forEach(
          replacements,
          ({ entry, client }) => Ref.set(entry.ref, client),
          { discard: true },
        )
        if (sourceChanged && nextSource !== "unavailable") {
          yield* publishToolAvailability(next)
        }
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

export function toolAvailabilityFromSource(
  source: WebSearchSource,
): ToolAvailabilityState {
  return {
    webSearch: source === "unavailable"
      ? { _tag: "Unavailable" }
      : { _tag: "Available", source },
  }
}

export function availabilitySource(
  state: ToolAvailabilityState,
): WebSearchSource {
  return state.webSearch._tag === "Available"
    ? state.webSearch.source
    : "unavailable"
}

export function nextToolAvailability(
  current: ToolAvailabilityState,
  source: WebSearchSource,
): ToolAvailabilityState {
  return availabilitySource(current) === source
    ? current
    : toolAvailabilityFromSource(source)
}
