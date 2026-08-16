import { describe, expect, test } from "vitest"
import { Option } from "effect"
import {
  PENTAGON_RADAR_DURATION_MS,
  interpolatePentagonRadar,
  pentagonRadarCell,
  pentagonRadarEase,
  pentagonRadarTransitionValues,
  renderPentagonRadar,
  retargetPentagonRadar,
  type PentagonRadarValues,
} from "../../components/pentagon-radar"
import { localModelRadarAxes } from "../local-inference/model-radar"
import { makeCatalogOnlyModel } from "../local-inference/test-fixtures"

const zero: PentagonRadarValues = [0, 0, 0, 0, 0].map(Option.some) as unknown as PentagonRadarValues
const one: PentagonRadarValues = [1, 1, 1, 1, 1].map(Option.some) as unknown as PentagonRadarValues

describe("catalog radar", () => {
  test("derives all five catalog axes and their displayed metrics", () => {
    const axes = Option.getOrThrow(localModelRadarAxes(makeCatalogOnlyModel()))
    expect(Option.getOrThrow(axes[0].value)).toBe(0.75)
    expect(Option.getOrThrow(axes[1].value)).toBeGreaterThanOrEqual(0)
    expect(Option.getOrThrow(axes[2].value)).toBe(0)
    expect(Option.getOrThrow(axes[3].value)).toBe(1)
    expect(Option.getOrThrow(axes[4].value)).toBe(0.75)
    expect(axes.map(({ label }) => label)).toEqual([
      "INTELLIGENCE",
      "SPEED",
      "SPECULATION",
      "MEMORY",
      "ACCURACY",
    ])
  })

  test("uses cubic ease-out and clamps interpolation", () => {
    expect(pentagonRadarEase(0)).toBe(0)
    expect(pentagonRadarEase(0.5)).toBeCloseTo(0.875)
    expect(pentagonRadarEase(1)).toBe(1)
    expect(interpolatePentagonRadar(zero, one, -1)).toEqual(zero)
    expect(interpolatePentagonRadar(zero, one, 2)).toEqual(one)
  })

  test("retargets through three and four profiles without snapping", () => {
    const first = { from: zero, to: one, startedAt: 100 }
    const halfway = pentagonRadarTransitionValues(first, 100 + PENTAGON_RADAR_DURATION_MS / 2)
    const third = [0.2, 0.6, 1, 0.4, 0.8].map(Option.some) as unknown as PentagonRadarValues
    const retargeted = retargetPentagonRadar(one, third, first, 100 + PENTAGON_RADAR_DURATION_MS / 2)
    expect(retargeted.from).toEqual(halfway)
    const fourth = [1, 0.6, 0.2, 0.8, 0.4].map(Option.some) as unknown as PentagonRadarValues
    const retargetedAgain = retargetPentagonRadar(third, fourth, retargeted, 180)
    expect(retargetedAgain.from).toEqual(pentagonRadarTransitionValues(retargeted, 180))
    expect(pentagonRadarTransitionValues(retargetedAgain, 180)).toEqual(retargetedAgain.from)
  })

  test("renders a full-width pentagon with every vertex metric", () => {
    const axes = Option.getOrThrow(localModelRadarAxes(makeCatalogOnlyModel()))
    const frame = renderPentagonRadar(axes)
    expect(frame).toHaveLength(15)
    const lines = frame.map((row) => row.map(({ text }) => text).join(""))
    expect(lines.every((line) => [...line].length === 56)).toBe(true)
    for (const { label, detail } of axes) {
      expect(lines.join("\n")).toContain(label)
      expect(lines.join("\n")).toContain(detail)
    }
  })

  test("never merges guide dots into profile-colored Braille cells", () => {
    expect(pentagonRadarCell(0x01, 0x80)).toEqual({
      character: String.fromCodePoint(0x2801),
      tone: "profile",
    })
    expect(pentagonRadarCell(0, 0x80)).toEqual({
      character: String.fromCodePoint(0x2880),
      tone: "grid",
    })
  })
})
