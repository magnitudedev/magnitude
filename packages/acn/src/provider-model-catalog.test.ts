import { Effect, Layer, Option, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelCatalogError,
  ModelDiscoveryOperationIdSchema,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  type ProviderCatalogOutcome,
  type ProviderClientShape,
  type ProviderModel,
} from "@magnitudedev/sdk"
import { LocalProviderOfferingProjection } from "./local-provider-offering-projection"
import { MirroredStateChangesLive } from "./mirrored-state"
import { ProviderModelCatalog, ProviderModelCatalogLive } from "./provider-model-catalog"
import { ProviderClient } from "@magnitudedev/sdk"

const providerA = ProviderIdSchema.make("provider-a")
const providerB = ProviderIdSchema.make("provider-b")
const effort = ReasoningEffortSchema.make("none")

const model = (providerId: typeof providerA, name: string): ProviderModel => ({
  providerId,
  providerModelId: ProviderModelIdSchema.make(`model-${name.toLowerCase()}`),
  displayName: `Model ${name}`,
  contextWindow: 8_192,
  maxOutputTokens: 1_024,
  defaultReasoningEffort: effort,
  properties: {
    vision: new VisionProperty.states.Resolved({ value: false }),
    reasoning: new ReasoningProperty.states.Resolved({ value: [effort] }),
  },
  servingCapabilities: { tools: true, structuredOutput: false },
  availability: { _tag: "Available" },
  pricing: { input: 0, output: 0, cached_input: null },
})

describe("provider model catalog", () => {
  it("retains failures for providers omitted by a targeted refresh", async () => {
    const failure = new ModelCatalogError({ message: "provider B unavailable" })
    const initial: readonly ProviderCatalogOutcome[] = [
      { _tag: "Success", providerId: providerA, models: [model(providerA, "A")] },
      { _tag: "Success", providerId: providerB, models: [model(providerB, "B")] },
    ]

    const state = await Effect.runPromise(Effect.gen(function* () {
      const outcomes = yield* Ref.make(initial)
      const client: ProviderClientShape = {
        catalog: {
          list: Effect.succeed([model(providerA, "A"), model(providerB, "B")]),
          refresh: Effect.succeed([model(providerA, "A"), model(providerB, "B")]),
          get: () => Effect.fail(new ModelCatalogError({ message: "not used" })),
        },
        catalogs: {
          list: Ref.get(outcomes),
          refresh: () => Ref.get(outcomes),
        },
        listProviders: Effect.succeed([
          { id: providerA, displayName: "Provider A", authStatus: { _tag: "authenticated" }, status: "ok" },
          { id: providerB, displayName: "Provider B", authStatus: { _tag: "authenticated" }, status: "error", message: failure.message },
        ]),
        sessionId: null,
        resolveModel: () => Effect.die("not used"),
        discoverModelProperties: () => Effect.succeed(ModelDiscoveryOperationIdSchema.make("not-used")),
        requestAttribution: (_providerId, _providerModelId, key) => ({ key, requestStarted: Effect.void }),
        webSearch: () => Effect.die("not used"),
        usage: () => Effect.die("not used"),
        runtimeConfig: { disableTraits: false },
      }
      const dependencies = Layer.mergeAll(
        Layer.succeed(ProviderClient, ProviderClient.of(client)),
        Layer.succeed(LocalProviderOfferingProjection, LocalProviderOfferingProjection.of({
          list: Effect.succeed([]),
          state: Effect.succeed({ entries: [], failure: Option.none() }),
          changes: Stream.empty,
        })),
        MirroredStateChangesLive,
      )
      return yield* Effect.gen(function* () {
        const catalog = yield* ProviderModelCatalog
        yield* Ref.set(outcomes, [
          { _tag: "Success", providerId: providerA, models: [model(providerA, "A")] },
          { _tag: "Failure", providerId: providerB, failure },
        ])
        yield* catalog.refresh(Option.none())
        yield* Ref.set(outcomes, [{ _tag: "Success", providerId: providerA, models: [model(providerA, "A")] }])
        yield* catalog.refresh(Option.some(providerA))
        return (yield* catalog.snapshot).state
      }).pipe(Effect.provide(ProviderModelCatalogLive.pipe(Layer.provide(dependencies))))
    }))

    expect(state._tag).toBe("Degraded")
    if (state._tag !== "Degraded") return
    expect(state.failures).toContainEqual({
      _tag: "ProviderFailure",
      providerId: providerB,
      message: failure.message,
    })
    expect(state.models.some((entry) => entry.providerId === providerB)).toBe(true)
    expect(state.models.find((entry) => entry.providerId === providerB)?.availability).toEqual({
      _tag: "Disabled",
      reason: "provider_unavailable",
    })
  })
})
