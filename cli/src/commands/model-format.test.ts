import { describe, expect, it } from "vitest"
import { modelContext } from "./model-format"
import { fitsAssessment, makeFoundModel } from "./fixtures/inference"

describe("model context column", () => {
  it("shows the context a model is served at, with its own when this computer serves it at less", () => {
    const served = (profile: number, context: number) => {
      const model = makeFoundModel()
      if (model.servingState._tag !== "Assessed") throw new Error("fixture is assessed")
      return makeFoundModel({ servingState: {
        ...model.servingState,
        assessment: { ...fitsAssessment(), profile: { contextLength: profile }, served: { contextLength: context } },
      } })
    }
    expect(modelContext(served(262_144, 65_536))).toBe("64K of 256K")
    expect(modelContext(served(262_144, 262_144))).toBe("256K")
    expect(modelContext(served(40_960, 40_960))).toBe("40K")
  })
})
