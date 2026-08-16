import { describe, expect, it } from "vitest"
import { getDisplayWidth } from "@magnitudedev/client-common"
import { formatModelClassification } from "./model-classification"

describe("model classification", () => {
  it("rounds catalog parameter facts to whole billions", () => {
    expect(formatModelClassification({
      architecture: "dense",
      totalParameters: 29_600_000_000,
    }, true)).toBe("Dense · 30B total parameters · Vision and text")
    expect(formatModelClassification({
      architecture: "mixtureOfExperts",
      totalParameters: 117_600_000_000,
      activeParameters: 8_500_000_000,
    }, false)).toBe("MoE · 118B total / 9B active parameters · Text only")
  })

  it("fits the longest current classification in 58 columns", () => {
    const classification = formatModelClassification({
      architecture: "mixtureOfExperts",
      totalParameters: 753_329_940_480,
      activeParameters: 40_000_000_000,
    }, true)

    expect(classification).toBe("MoE · 753B total / 40B active parameters · Vision and text")
    expect(getDisplayWidth(classification)).toBeLessThanOrEqual(58)
  })
})
