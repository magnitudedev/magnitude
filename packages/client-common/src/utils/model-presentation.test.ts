import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { ModelReleaseDateSchema, ModelVariantLabelSchema } from "@magnitudedev/sdk"
import {
  formatLocalModelDisplayName,
  formatModelDisplayName,
  formatModelParameters,
  formatModelReleaseAge,
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
    expect(formatSpeculativeMethod({ _tag: "LiLiCorr" })).toBe("LiLiCorr")
  })
})

describe("model release age and size", () => {
  it("counts whole local calendar days since release", () => {
    expect(formatModelReleaseAge(ModelReleaseDateSchema.make("2026-08-13"), new Date(2026, 7, 16, 23, 59))).toBe("3 days ago")
    expect(formatModelReleaseAge(ModelReleaseDateSchema.make("2026-08-15"), new Date(2026, 7, 16, 0, 1))).toBe("1 day ago")
    expect(formatModelReleaseAge(ModelReleaseDateSchema.make("2026-08-16"), new Date(2026, 7, 16, 12))).toBe("Today")
  })
  it("states a dense model's size and a mixture of experts' total and active parameters", () => {
    expect(formatModelParameters({ architecture: "dense", totalParameters: 27_000_000_000 })).toBe("27B")
    expect(formatModelParameters({ architecture: "dense", totalParameters: 2_600_000_000 })).toBe("2.6B")
    expect(formatModelParameters({ architecture: "mixtureOfExperts", totalParameters: 26_000_000_000, activeParameters: 4_000_000_000 })).toBe("26B total · 4B active")
  })
})
