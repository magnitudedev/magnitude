import { describe, expect, it } from "vitest"
import { Effect, Ref } from "effect"
import { FetchHttpClient } from "@effect/platform"
import type { ProviderAuth } from "@magnitudedev/protocol"
import { ModelDiscoveryOperationIdSchema, ModelFamilyIdSchema, ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema, ReasoningProperty, VisionProperty, type ProviderClientShape } from "@magnitudedev/sdk"
import {
  makeDelegatingProviderClient,
  resolveEndpointProviderAuthFromStorage,
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

const providerClient = (label: string): ProviderClientShape => ({
  catalog: {
    list: Effect.succeed([{
      providerId: ProviderIdSchema.make(label),
      providerModelId: ProviderModelIdSchema.make("model"),
      modelFamilyId: ModelFamilyIdSchema.make("family"),
      displayName: label,
      contextWindow: 1,
      maxOutputTokens: 1,
      defaultReasoningEffort: ReasoningEffortSchema.make("none"),
      properties: {
        vision: new VisionProperty.states.Resolved({ value: false }),
        reasoning: new ReasoningProperty.states.Resolved({ value: [ReasoningEffortSchema.make("none")] }),
      },
      availability: { _tag: "Available" },
      pricing: { input: 0, output: 0, cached_input: null },
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
  requestAttribution: () => ({ requestStarted: Effect.void }),
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
      yield* Ref.set(ref, providerClient("second"))
      expect((yield* stable.catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))[0]?.providerId).toBe("second")
      expect(stable.sessionId).toBe("session")
    }))
  })
})
