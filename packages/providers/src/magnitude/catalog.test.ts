import { describe, expect, it } from "vitest"
import { toMagnitudeModelInfo } from "./catalog"
import type { MagnitudeRawModel } from "./contract"

const rawModel = (overrides: Partial<MagnitudeRawModel> = {}): MagnitudeRawModel => ({
  id: "test-model",
  object: "model",
  owned_by: "magnitude",
  displayName: "Test Model",
  roles: ["leader"],
  slots: ["primary"],
  contextWindow: 200_000,
  maxOutputTokens: 128_000,
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
    const model = toMagnitudeModelInfo(rawModel({ id: "another-model" }))

    expect(model.properties.reasoning).toMatchObject({
      _tag: "Resolved",
      value: ["none", "low", "medium", "high", "max"],
    })
  })
})
