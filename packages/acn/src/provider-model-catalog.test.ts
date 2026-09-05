import { Deferred, Effect, Fiber, Layer, Option, Ref, Stream } from "effect"
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
} from "@magnitudedev/providers/client"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { AcnChangesLive } from "./changes"
import { ProviderModelCatalog, ProviderModelCatalogLive } from "./provider-model-catalog"
import { ProviderClient } from "@magnitudedev/providers/client"

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
  pricing: Option.some({ input: 0, output: 0, cached_input: null }),
})

describe("provider model catalog", () => {
  it("degrades an empty catalog and retains failures for providers omitted by a targeted refresh", async () => {
    const failure = new ModelCatalogError({ message: "provider B unavailable" })
    const initialFailures: readonly ProviderCatalogOutcome[] = [
      { _tag: "Failure", providerId: providerA, failure },
      { _tag: "Failure", providerId: providerB, failure },
    ]
    const available: readonly ProviderCatalogOutcome[] = [
      { _tag: "Success", providerId: providerA, models: [model(providerA, "A")] },
      { _tag: "Success", providerId: providerB, models: [model(providerB, "B")] },
    ]

    const result = await Effect.runPromise(Effect.gen(function* () {
      const outcomes = yield* Ref.make(initialFailures)
      const refreshCalls = yield* Ref.make(0)
      const refreshEntered = yield* Deferred.make<void>()
      const releaseRefresh = yield* Deferred.make<void>()
      const pauseRefresh = yield* Ref.make(false)
      const defectRefresh = yield* Ref.make(false)
      const client: ProviderClientShape = {
        catalog: {
          list: Effect.succeed([model(providerA, "A"), model(providerB, "B")]),
          refresh: Effect.succeed([model(providerA, "A"), model(providerB, "B")]),
          get: () => Effect.fail(new ModelCatalogError({ message: "not used" })),
        },
        catalogs: {
          list: Ref.get(outcomes),
          refresh: () => Effect.gen(function* () {
            yield* Ref.update(refreshCalls, (count) => count + 1)
            if (yield* Ref.get(defectRefresh)) {
              return yield* Effect.die("provider refresh defect")
            }
            if (yield* Ref.get(pauseRefresh)) {
              yield* Deferred.succeed(refreshEntered, undefined)
              yield* Deferred.await(releaseRefresh)
            }
            return yield* Ref.get(outcomes)
          }),
        },
        listProviders: Effect.succeed([
          { id: providerA, displayName: "Provider A", kind: "Hosted", authStatus: { _tag: "authenticated" }, status: "ok" },
          { id: providerB, displayName: "Provider B", kind: "Hosted", authStatus: { _tag: "authenticated" }, status: "error", message: failure.message },
        ]),
        sessionId: null,
        resolveModel: () => Effect.die("not used"),
        discoverModelProperties: () => Effect.succeed(ModelDiscoveryOperationIdSchema.make("not-used")),
        requestAttribution: (_providerId, _providerModelId, key) => ({ key, requestStarted: Effect.void }),
        webSearchSource: Effect.succeed("unavailable"),
        webSearch: () => Effect.die("not used"),
        usage: () => Effect.die("not used"),
        runtimeConfig: { disableTraits: false },
      }
      const dependencies = Layer.mergeAll(
        Layer.succeed(ProviderClient, ProviderClient.of(client)),
        Layer.succeed(LocalProviderOfferings, LocalProviderOfferings.of({
          ready: Effect.succeed(true),
          list: Effect.succeed([]),
          catalog: Effect.succeed([]),
          changes: Stream.never,
          catalogChanges: Stream.never,
          resolve: () => Effect.die("unused"),
        })),
        AcnChangesLive,
      )
      return yield* Effect.gen(function* () {
        const catalog = yield* ProviderModelCatalog
        const initialState = yield* catalog.state
        yield* Ref.set(outcomes, available)
        yield* catalog.refresh(Option.none())
        yield* Ref.set(outcomes, [
          { _tag: "Success", providerId: providerA, models: [model(providerA, "A")] },
          { _tag: "Failure", providerId: providerB, failure },
        ])
        const refreshCallsBeforeConcurrency = yield* Ref.get(refreshCalls)
        yield* Ref.set(pauseRefresh, true)
        const firstCaller = yield* catalog.refresh(Option.none()).pipe(Effect.fork)
        yield* Deferred.await(refreshEntered)
        yield* Fiber.interrupt(firstCaller)
        const joiningCaller = yield* catalog.refresh(Option.none()).pipe(Effect.fork)
        const conflictingCaller = yield* catalog.refresh(Option.some(providerA)).pipe(Effect.fork)
        yield* Deferred.succeed(releaseRefresh, undefined)
        yield* Fiber.join(joiningCaller)
        yield* Fiber.join(conflictingCaller)
        expect(yield* Ref.get(refreshCalls)).toBe(refreshCallsBeforeConcurrency + 2)
        yield* Ref.set(pauseRefresh, false)
        yield* Ref.set(outcomes, [{ _tag: "Success", providerId: providerA, models: [model(providerA, "A")] }])
        yield* catalog.refresh(Option.some(providerA))
        const refreshCount = yield* Ref.get(refreshCalls)
        yield* catalog.state
        expect(yield* Ref.get(refreshCalls)).toBe(refreshCount)
        yield* Ref.set(defectRefresh, true)
        yield* catalog.refresh(Option.some(providerB))
        return {
          initialState,
          finalState: yield* catalog.state,
        }
      }).pipe(Effect.provide(ProviderModelCatalogLive.pipe(Layer.provide(dependencies))))
    }))

    expect(result.initialState._tag).toBe("Degraded")
    if (result.initialState._tag !== "Degraded") return
    expect(result.initialState.models).toEqual([])
    expect(result.initialState.failures).toHaveLength(2)

    const state = result.finalState
    expect(state._tag).toBe("Degraded")
    if (state._tag !== "Degraded") return
    expect(state.failures).toContainEqual({
      _tag: "ProviderFailure",
      providerId: providerB,
      message: failure.message,
    })
    expect(state.failures.some((entry) => entry._tag === "CatalogFailure")).toBe(true)
    expect(state.models.some((entry) => entry.providerId === providerB)).toBe(true)
    expect(state.models.find((entry) => entry.providerId === providerB)?.availability).toEqual({
      _tag: "Disabled",
      reason: "provider_unavailable",
    })
  })
})
