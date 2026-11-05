import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { ModelVariantLabelSchema, type CustomLocalModel, type SpeculativeMethod } from "@magnitudedev/sdk"
import { makeFoundModel, makeSetupModel, streamedFitResidency } from "../application/fixtures/model"
import { localModelRadarAxes, localModelSpeedTooltip, normalizeLocalModelRadarSpeed } from "./local-model-radar"

const axis = (model: Parameters<typeof localModelRadarAxes>[0], label: string) =>
  Option.getOrThrow(localModelRadarAxes(model)).find((candidate) => candidate.label === label)!

const withQuantization = (quantization: string, quantizationName: string, variantLabel = quantizationName): CustomLocalModel => {
  const model = makeFoundModel()
  if (model.servingState._tag !== "Assessed") throw new Error("fixture is assessed")
  return makeFoundModel({
    presentation: { ...model.presentation, variantLabel: ModelVariantLabelSchema.make(variantLabel) },
    servingState: { ...model.servingState, metadata: { ...model.servingState.metadata, quantization, quantizationName } },
  })
}

const speculating = (speculativeMethod: Option.Option<SpeculativeMethod>): CustomLocalModel => {
  const model = makeFoundModel()
  if (model.servingState._tag !== "Assessed") throw new Error("fixture is assessed")
  return makeFoundModel({ servingState: { ...model.servingState, speculativeMethod } })
}

describe("local model radar", () => {
  it("says which context depths the speed range is estimated at", () => {
    expect(localModelSpeedTooltip).toContain("25K–75K tokens of context")
  })

  it("never rates a custom model's intelligence", () => {
    const intelligence = axis(makeFoundModel(), "INTELLIGENCE")
    expect(intelligence.value).toEqual(Option.none())
    expect(intelligence.detail).toBe("Not rated")
  })

  it("rates a custom model's fidelity from its weight type on the catalog's scale", () => {
    expect(axis(makeFoundModel(), "FIDELITY")).toEqual({ label: "FIDELITY", value: Option.some(0.4), detail: "Medium (Q4)" })
    expect(axis(withQuantization("Q8_0", "Q8_0"), "FIDELITY")).toEqual({ label: "FIDELITY", value: Option.some(0.8), detail: "Very high (Q8)" })
    expect(axis(withQuantization("Q5_K", "Q5_K_M"), "FIDELITY").detail).toBe("High (Q5)")
    expect(axis(withQuantization("Q4_K", "4-bit", "Q4_K_M"), "FIDELITY").detail).toBe("Medium (Q4)")
    expect(axis(withQuantization("unknown", "custom"), "FIDELITY")).toEqual({ label: "FIDELITY", value: Option.none(), detail: "Not rated (custom)" })
  })

  it("estimates a custom model's speed, fit, and speculation like a catalog model's", () => {
    const custom = makeFoundModel()
    expect(axis(custom, "SPEED").value).toEqual(Option.some(normalizeLocalModelRadarSpeed(40)))
    expect(axis(custom, "FIT").detail).toMatch(/^Spacious/)
    expect(axis(custom, "SPECULATION").detail).toBe("None")
    expect(axis(speculating(Option.some({ _tag: "DFlash" })), "SPECULATION").value).toEqual(Option.some(0.4))
  })

  it("rates a streamed fit's speed at its allowed share, and no speed while it is unmeasured", () => {
    const serving = makeFoundModel().servingState
    if (serving._tag !== "Assessed" || serving.assessment._tag !== "Fits") throw new Error("fixture fits")
    const fits = serving.assessment
    const streamed = (unmeasured: boolean) => makeFoundModel({
      servingState: { ...serving, assessment: { ...fits, residency: streamedFitResidency(unmeasured) } },
    })
    expect(axis(streamed(false), "SPEED")).toMatchObject({ value: Option.some(normalizeLocalModelRadarSpeed(20.4)), detail: "~20 tok/s" })
    expect(axis(streamed(true), "SPEED")).toMatchObject({ value: Option.none(), detail: "Measured on its first streamed load" })
  })

  it("spaces speculative methods evenly from none to the strongest", () => {
    const ladder = [Option.none(), ...(["Mtp", "DFlash", "DSpark", "DFlash2", "LiLiCorr"] as const)
      .map((_tag) => Option.some({ _tag }))]
      .map((method) => axis(speculating(method), "SPECULATION").value)
    expect(ladder).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1].map(Option.some))
  })

  it("keeps a catalog model's intelligence and catalog fidelity rank", () => {
    expect(axis(makeSetupModel(true), "INTELLIGENCE").detail).toBe("1%")
    expect(axis(makeSetupModel(true), "FIDELITY").detail).toBe("Reduced (Q4)")
  })
})
