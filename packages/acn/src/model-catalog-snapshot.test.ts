import { describe, expect, it } from "vitest"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  type ProviderModel,
} from "@magnitudedev/sdk"
import { ModelCatalogError } from "@magnitudedev/sdk"
import { foldProviderCatalogOutcomes } from "./model-catalog-snapshot"

const model = (providerId: string, providerModelId: string): ProviderModel => ({
  providerId: ProviderIdSchema.make(providerId),
  providerModelId: ProviderModelIdSchema.make(providerModelId),
  displayName: providerModelId,
  contextWindow: 8_192,
  maxOutputTokens: 1_024,
  defaultReasoningEffort: ReasoningEffortSchema.make("none"),
  properties: {
    vision: new VisionProperty.states.Resolved({ value: false }),
    reasoning: new ReasoningProperty.states.Resolved({ value: [ReasoningEffortSchema.make("none")] }),
  },
  availability: { _tag: "Available" },
  pricing: { input: 0, output: 0, cached_input: null },
})

describe("provider catalog outcomes", () => {
  it("updates healthy providers while retaining failed providers", () => {
    const cloud = ProviderIdSchema.make("cloud")
    const local = ProviderIdSchema.make("local")
    const previous = {
      byProvider: new Map([
        [cloud, [model("cloud", "old-cloud")]],
        [local, [model("local", "old-local")]],
      ]),
      failuresByProvider: new Map(),
    }

    const result = foldProviderCatalogOutcomes(previous, [
      { _tag: "Failure", providerId: cloud, failure: new ModelCatalogError({ message: "offline" }) },
      { _tag: "Success", providerId: local, models: [model("local", "new-local")] },
    ])

    expect(result.models).toEqual([
      model("cloud", "old-cloud"),
      model("local", "new-local"),
    ])
    expect(result.failures).toMatchObject([{ _tag: "stale", providerId: cloud, message: "offline" }])
  })

  it("treats an empty successful provider result as authoritative", () => {
    const cloud = ProviderIdSchema.make("cloud")
    const local = ProviderIdSchema.make("local")
    const previous = {
      byProvider: new Map([
        [cloud, [model("cloud", "old-cloud")]],
        [local, [model("local", "old-local")]],
      ]),
      failuresByProvider: new Map(),
    }

    const result = foldProviderCatalogOutcomes(previous, [
      { _tag: "Failure", providerId: cloud, failure: new ModelCatalogError({ message: "offline" }) },
      { _tag: "Success", providerId: local, models: [] },
    ])

    expect(result.models).toEqual([model("cloud", "old-cloud")])
    expect(result.failures[0]?._tag).toBe("stale")
  })

  it("marks a failed provider without retained models unavailable", () => {
    const providerId = ProviderIdSchema.make("cloud")
    const result = foldProviderCatalogOutcomes({ byProvider: new Map(), failuresByProvider: new Map() }, [{
      _tag: "Failure",
      providerId,
      failure: new ModelCatalogError({ message: "offline" }),
    }])

    expect(result.models).toEqual([])
    expect(result.failures).toMatchObject([{ _tag: "unavailable", providerId, message: "offline" }])
  })

  it("retains failures for providers not included in a targeted refresh", () => {
    const cloud = ProviderIdSchema.make("cloud")
    const local = ProviderIdSchema.make("local")
    const failed = foldProviderCatalogOutcomes({ byProvider: new Map(), failuresByProvider: new Map() }, [{
      _tag: "Failure",
      providerId: cloud,
      failure: new ModelCatalogError({ message: "offline" }),
    }])

    const refreshed = foldProviderCatalogOutcomes(failed, [{
      _tag: "Success",
      providerId: local,
      models: [model("local", "new-local")],
    }])

    expect(refreshed.failures).toMatchObject([{ _tag: "unavailable", providerId: cloud }])
  })
})
