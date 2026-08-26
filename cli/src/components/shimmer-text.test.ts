import { describe, expect, it } from "vitest"
import { sweepShimmerIntensities } from "./shimmer-text"

describe("sweep shimmer", () => {
  it("is fully absent during the dwell between sweeps", () => {
    expect(sweepShimmerIntensities({ length: 12, timeMs: 1_200 }))
      .toEqual(Array.from({ length: 12 }, () => 0))
  })

  it("uses a smooth localized peak instead of a full-width linear gradient", () => {
    const intensities = sweepShimmerIntensities({
      length: 9,
      timeMs: 425,
      sweepDurationMs: 850,
      cycleDurationMs: 1_800,
    })
    expect(intensities[4]).toBeCloseTo(1)
    expect(intensities[3]).toBeGreaterThan(intensities[2]!)
    expect(intensities[5]).toBeGreaterThan(intensities[6]!)
    expect(intensities[0]).toBe(0)
    expect(intensities[8]).toBe(0)
  })

  it("widens the streak in proportion to the text length", () => {
    const litCharacterCount = (length: number) => sweepShimmerIntensities({
      length,
      timeMs: 425,
    }).filter((intensity) => intensity > 0).length

    expect(litCharacterCount(30)).toBeGreaterThan(litCharacterCount(10))
  })

  it("eases the sweep position rather than moving it linearly", () => {
    const quarterSweep = sweepShimmerIntensities({
      length: 20,
      timeMs: 212.5,
      sweepDurationMs: 850,
      cycleDurationMs: 1_800,
      radius: 2,
    })
    const brightestIndex = quarterSweep.indexOf(Math.max(...quarterSweep))
    expect(brightestIndex).toBeLessThan(5)
  })
})
