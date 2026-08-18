import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { ModelVariantLabelSchema } from "@magnitudedev/sdk"
import {
  formatLocalModelDisplayName,
  formatModelDisplayName,
  formatSpeculativeMethod,
} from "./model-presentation"

describe("model presentation", () => {
  it("appends a structured variant label", () => {
    expect(formatModelDisplayName(
      "Nemotron 3.5 Lightning 30B-A3B",
      Option.some(ModelVariantLabelSchema.make("NVFP4")),
    )).toBe("Nemotron 3.5 Lightning 30B-A3B (NVFP4)")
  })

  it("leaves models without a variant label unchanged", () => {
    expect(formatModelDisplayName("Cloud Model", Option.none())).toBe("Cloud Model")
  })

  it("formats local presentation without inspecting artifact metadata", () => {
    expect(formatLocalModelDisplayName({
      presentation: {
        displayName: "Gemma 4 12B",
        variantLabel: ModelVariantLabelSchema.make("Q4 QAT"),
      },
    })).toBe("Gemma 4 12B (Q4 QAT)")
  })

  it("formats speculative methods consistently", () => {
    expect(formatSpeculativeMethod({ _tag: "Mtp" })).toBe("MTP")
    expect(formatSpeculativeMethod({ _tag: "DFlash" })).toBe("DFlash")
    expect(formatSpeculativeMethod({ _tag: "DSpark" })).toBe("DSpark")
  })
})
