import { describe, expect, it } from "vitest"
import { getDisplayWidth } from "@magnitudedev/client-common"
import type { ModelReleaseDate } from "@magnitudedev/sdk"
import { formatModelClassification, formatModelReleaseRecency } from "./model-classification"

describe("model classification", () => {
  it("rounds catalog parameter facts to whole billions", () => {
    expect(formatModelClassification({
      architecture: "dense",
      totalParameters: 29_600_000_000,
    }, true)).toBe("Dense (30B) · Text and vision")
    expect(formatModelClassification({
      architecture: "mixtureOfExperts",
      totalParameters: 117_600_000_000,
      activeParameters: 8_500_000_000,
    }, false)).toBe("MoE (118B / 9B) · Text only")
  })

  it("fits the longest current classification in 58 columns", () => {
    const classification = formatModelClassification({
      architecture: "mixtureOfExperts",
      totalParameters: 753_329_940_480,
      activeParameters: 40_000_000_000,
    }, true)

    expect(classification).toBe("MoE (753B / 40B) · Text and vision")
    expect(getDisplayWidth(classification)).toBeLessThanOrEqual(58)
  })

  it("reports release recency as rounded local calendar days", () => {
    expect(formatModelReleaseRecency(
      "2026-08-13" as ModelReleaseDate,
      new Date(2026, 7, 16, 23, 59),
    )).toBe("3 days ago")
    expect(formatModelReleaseRecency(
      "2026-08-15" as ModelReleaseDate,
      new Date(2026, 7, 16, 0, 1),
    )).toBe("1 day ago")
    expect(formatModelReleaseRecency(
      "2026-03-16" as ModelReleaseDate,
      new Date(2026, 7, 16, 12),
    )).toBe("153 days ago")
  })
})
