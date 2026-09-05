import { describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Option, Ref } from "effect"
import { FetchHttpClient } from "@effect/platform"
import type { ProviderAuth } from "@magnitudedev/acn-protocol"
import {
  ModelDiscoveryOperationIdSchema,
  ModelFamilyIdSchema,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  type ProviderClientShape,
} from "@magnitudedev/providers/client"
import {
  nextToolAvailability,
  makeDelegatingProviderClient,
  resolveEndpointProviderAuthFromStorage,
  toolAvailabilityFromSource,
} from "./shared-client"

function storageWithAuth(initial?: ProviderAuth) {
  const entries: Record<string, ProviderAuth> = initial ? { local: initial } : {}
  const storage = {
    auth: {
      get: (providerId: string) => Effect.succeed(entries[providerId]),
    },
  }
  return { entries, storage }
}

describe("endpoint provider auth resolution", () => {
  it("returns a missing provider's default endpoint without mutating auth storage", async () => {
    const { entries, storage } = storageWithAuth()

    const resolved = await Effect.runPromise(
      resolveEndpointProviderAuthFromStorage(storage, "local", {
        endpoint: "http://127.0.0.1:8080",
      }),
    )

    expect(resolved).toEqual({ endpoint: "http://127.0.0.1:8080" })
    expect(entries.local).toBeUndefined()
  })

  it("uses an explicit endpoint instead of the default", async () => {
    const { entries, storage } = storageWithAuth({
      type: "endpoint",
      endpoint: "http://127.0.0.1:9090",
    })

    const resolved = await Effect.runPromise(
      resolveEndpointProviderAuthFromStorage(storage, "local", {
        endpoint: "http://127.0.0.1:8080",
      }),
    )

    expect(resolved).toEqual({ endpoint: "http://127.0.0.1:9090" })
    expect(entries.local).toEqual({
      type: "endpoint",
      endpoint: "http://127.0.0.1:9090",
    })
  })
})

const providerClient = (
  label: string,
  webSearchSource: "magnitude" | "exa" | "unavailable" = "unavailable",
): ProviderClientShape => ({
  catalog: {
    list: Effect.succeed([{
      providerId: ProviderIdSchema.make(label),
      providerModelId: ProviderModelIdSchema.make("model"),
      modelFamilyId: ModelFamilyIdSchema.make("family"),
      displayName: label,
      contextWindow: 1,
      maxOutputTokens: 1,
      defaultReasoningEffort: ReasoningEffortSchema.make("none"),
      servingCapabilities: { tools: true, structuredOutput: false },
      properties: {
        vision: new VisionProperty.states.Resolved({ value: false }),
        reasoning: new ReasoningProperty.states.Resolved({ value: [ReasoningEffortSchema.make("none")] }),
      },
      availability: { _tag: "Available" },
      pricing: Option.some({ input: 0, output: 0, cached_input: null }),
    }]),
    get: () => Effect.die("not used"),
    refresh: Effect.die("not used"),
  },
  catalogs: {
    list: Effect.succeed([]),
    refresh: () => Effect.succeed([]),
  },
  listProviders: Effect.succeed([]),
  sessionId: "session",
  resolveModel: () => Effect.die("not used"),
  discoverModelProperties: () => Effect.succeed(ModelDiscoveryOperationIdSchema.make("test")),
  requestAttribution: (_providerId, _providerModelId, key) => ({ key, requestStarted: Effect.void }),
  webSearchSource: Effect.succeed(webSearchSource),
  webSearch: () => Effect.die("not used"),
  usage: () => Effect.die("not used"),
  runtimeConfig: { disableTraits: false },
})

describe("delegating provider client", () => {
  it("keeps client identity stable while resolving calls through a replacement", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const ref = yield* Ref.make(providerClient("first"))
      const stable = makeDelegatingProviderClient(ref, { disableTraits: false }, "session")

      expect((yield* stable.catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))[0]?.providerId).toBe("first")
      expect(yield* stable.webSearchSource).toBe("unavailable")
      yield* Ref.set(ref, providerClient("second", "exa"))
      expect((yield* stable.catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))[0]?.providerId).toBe("second")
      expect(yield* stable.webSearchSource).toBe("exa")
      expect(stable.sessionId).toBe("session")
    }))
  })

  it("keeps an in-flight web search on the concrete client that started it", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const started = yield* Deferred.make<void>()
      const release = yield* Deferred.make<void>()
      const first: ProviderClientShape = {
        ...providerClient("first", "magnitude"),
        webSearch: () => Effect.gen(function* () {
          yield* Deferred.succeed(started, undefined)
          yield* Deferred.await(release)
          return { text: "first", sources: [] }
        }),
      }
      const second: ProviderClientShape = {
        ...providerClient("second", "exa"),
        webSearch: () => Effect.succeed({ text: "second", sources: [] }),
      }
      const ref = yield* Ref.make(first)
      const stable = makeDelegatingProviderClient(ref, { disableTraits: false }, "session")

      const inFlight = yield* Effect.fork(stable.webSearch("query"))
      yield* Deferred.await(started)
      yield* Ref.set(ref, second)
      yield* Deferred.succeed(release, undefined)

      expect((yield* Fiber.join(inFlight)).text).toBe("first")
      expect((yield* stable.webSearch("query")).text).toBe("second")
    }).pipe(Effect.provide(FetchHttpClient.layer)))
  })
})

describe("tool availability values", () => {
  it("changes only when the semantic web-search source changes", () => {
    const initial = toolAvailabilityFromSource("unavailable")
    const unchanged = nextToolAvailability(initial, "unavailable")
    const exa = nextToolAvailability(unchanged, "exa")
    const cloud = nextToolAvailability(exa, "magnitude")

    expect(unchanged).toBe(initial)
    expect(exa).toEqual({
      webSearch: { _tag: "Available", source: "exa" },
    })
    expect(cloud).toEqual({
      webSearch: { _tag: "Available", source: "magnitude" },
    })
  })
})
