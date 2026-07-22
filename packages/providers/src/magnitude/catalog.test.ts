import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { ProviderModelIdSchema } from "@magnitudedev/ai"
import { toMagnitudeModelInfo } from "./catalog"
import type { MagnitudeRawModel } from "./contract"

const rawModel = (overrides: Partial<MagnitudeRawModel> = {}): MagnitudeRawModel => ({
  id: ProviderModelIdSchema.make("test-model"),
  object: "model",
  owned_by: "magnitude",
  displayName: "Test Model",
  roles: ["leader"],
  slots: ["primary"],
  tiers: Option.none(),
  type: Option.none(),
  contextWindow: 200_000,
  maxOutputTokens: 128_000,
  capabilities: Option.none(),
  pricing: Option.none(),
  ...overrides,
})

describe("Magnitude model catalog mapping", () => {
  it("assigns the provider-wide reasoning efforts without model-list metadata", () => {
    const model = toMagnitudeModelInfo(rawModel())

    expect(model.properties.reasoning).toMatchObject({
      _tag: "Resolved",
      value: ["none", "low", "medium", "high", "max"],
    })
  })

  it("assigns the same reasoning contract to every cloud model", () => {
    const model = toMagnitudeModelInfo(rawModel({ id: ProviderModelIdSchema.make("another-model") }))

    expect(model.properties.reasoning).toMatchObject({
      _tag: "Resolved",
      value: ["none", "low", "medium", "high", "max"],
    })
  })
})
