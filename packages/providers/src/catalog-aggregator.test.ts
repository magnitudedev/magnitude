import { describe, expect, it } from "vitest"
import { Cause, Chunk, Effect } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { ModelCatalogError, ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema, ReasoningProperty, VisionProperty, type ModelCatalog, type ProviderModel } from "@magnitudedev/ai"
import { inspectProviderCatalogs, makeAggregatedCatalog } from "./catalog-aggregator"

const catalog = (list: ModelCatalog<ProviderModel>["list"]): ModelCatalog<ProviderModel> => ({
  list,
  refresh: list,
  get: () => Effect.fail(new ModelCatalogError({ message: "not found" })),
})

describe("aggregated model catalog", () => {
  it("represents expected provider failures as outcomes", async () => {
    const failure = new ModelCatalogError({ message: "provider unavailable" })
    const providers = [{
      id: ProviderIdSchema.make("test"),
      catalog: catalog(Effect.fail(failure)),
    }]

    const result = await Effect.runPromise(inspectProviderCatalogs(providers, "list").pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result).toEqual([{
      _tag: "Failure",
      providerId: ProviderIdSchema.make("test"),
      failure,
    }])
  })

  it("preserves a successful authoritative empty catalog", async () => {
    const aggregated = makeAggregatedCatalog([{
      id: ProviderIdSchema.make("test"),
      catalog: catalog(Effect.succeed([])),
    }])

    await expect(Effect.runPromise(aggregated.list.pipe(Effect.provide(FetchHttpClient.layer)))).resolves.toEqual([])
  })

  it("keeps successful aggregate results while preserving failed-provider outcomes", async () => {
    const model: ProviderModel = {
      providerId: ProviderIdSchema.make("healthy"),
      providerModelId: ProviderModelIdSchema.make("model"),
      displayName: "Healthy model",
      contextWindow: 8_192,
      maxOutputTokens: 1_024,
      defaultReasoningEffort: ReasoningEffortSchema.make("none"),
      properties: {
        vision: new VisionProperty.states.Resolved({ value: false }),
        reasoning: new ReasoningProperty.states.Resolved({ value: [ReasoningEffortSchema.make("none")] }),
      },
      servingCapabilities: { tools: true, structuredOutput: false },
      availability: { _tag: "Available" },
      pricing: { input: 0, output: 0, cached_input: null },
    }
    const providers = [
      { id: ProviderIdSchema.make("failed"), catalog: catalog(Effect.fail(new ModelCatalogError({ message: "offline" }))) },
      { id: ProviderIdSchema.make("healthy"), catalog: catalog(Effect.succeed([model])) },
    ]
    const aggregated = makeAggregatedCatalog(providers)

    const [models, outcomes] = await Effect.runPromise(Effect.all([
      aggregated.list,
      inspectProviderCatalogs(providers, "list"),
    ]).pipe(Effect.provide(FetchHttpClient.layer)))

    expect(models).toEqual([model])
    expect(outcomes.map((outcome) => outcome._tag)).toEqual(["Failure", "Success"])
  })

  it("uses the same outcome semantics when refreshing", async () => {
    const failure = new ModelCatalogError({ message: "refresh unavailable" })
    const providers = [{
      id: ProviderIdSchema.make("test"),
      catalog: catalog(Effect.fail(failure)),
    }]

    const result = await Effect.runPromise(inspectProviderCatalogs(providers, "refresh").pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result[0]).toMatchObject({
      _tag: "Failure",
      providerId: ProviderIdSchema.make("test"),
      failure,
    })
  })

  it("does not convert defects into provider failure outcomes", async () => {
    const providers = [{
      id: ProviderIdSchema.make("defective"),
      catalog: catalog(Effect.die("broken invariant")),
    }]

    const exit = await Effect.runPromise(Effect.exit(
      inspectProviderCatalogs(providers, "list").pipe(Effect.provide(FetchHttpClient.layer)),
    ))

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      expect(Chunk.toReadonlyArray(Cause.defects(exit.cause))).toEqual(["broken invariant"])
    }
  })
})
