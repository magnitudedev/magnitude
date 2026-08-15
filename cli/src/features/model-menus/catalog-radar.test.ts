import { describe, expect, test } from "vitest"
import { Option } from "effect"
import {
  CATALOG_RADAR_DURATION_MS,
  catalogRadarCell,
  catalogRadarEase,
  catalogRadarProfile,
  catalogRadarTransitionValues,
  interpolateCatalogRadar,
  renderCatalogRadar,
  retargetCatalogRadar,
  type CatalogRadarValues,
} from "./catalog-radar"
import { makeCatalogOnlyModel } from "../local-inference/test-fixtures"

const zero: CatalogRadarValues = [0, 0, 0, 0, 0]
const one: CatalogRadarValues = [1, 1, 1, 1, 1]

describe("catalog radar", () => {
  test("derives all five catalog axes and their displayed metrics", () => {
    const profile = Option.getOrThrow(catalogRadarProfile(makeCatalogOnlyModel()))
    expect(profile.values[0]).toBe(0.75)
    expect(profile.values[1]).toBeGreaterThanOrEqual(0)
    expect(profile.values[2]).toBe(0)
    expect(profile.values[3]).toBe(1)
    expect(profile.values[4]).toBe(0.75)
    expect(profile.metrics.map(({ name }) => name)).toEqual([
      "INTELLIGENCE",
      "SPEED",
      "SPECULATION",
      "MEMORY",
      "ACCURACY",
    ])
  })

  test("uses cubic ease-out and clamps interpolation", () => {
    expect(catalogRadarEase(0)).toBe(0)
    expect(catalogRadarEase(0.5)).toBeCloseTo(0.875)
    expect(catalogRadarEase(1)).toBe(1)
    expect(interpolateCatalogRadar(zero, one, -1)).toEqual(zero)
    expect(interpolateCatalogRadar(zero, one, 2)).toEqual(one)
  })

  test("retargets through three and four profiles without snapping", () => {
    const first = { from: zero, to: one, startedAt: 100 }
    const halfway = catalogRadarTransitionValues(first, 100 + CATALOG_RADAR_DURATION_MS / 2)
    const third: CatalogRadarValues = [0.2, 0.6, 1, 0.4, 0.8]
    const retargeted = retargetCatalogRadar(one, third, first, 100 + CATALOG_RADAR_DURATION_MS / 2)
    expect(retargeted.from).toEqual(halfway)
    const fourth: CatalogRadarValues = [1, 0.6, 0.2, 0.8, 0.4]
    const retargetedAgain = retargetCatalogRadar(third, fourth, retargeted, 180)
    expect(retargetedAgain.from).toEqual(catalogRadarTransitionValues(retargeted, 180))
    expect(catalogRadarTransitionValues(retargetedAgain, 180)).toEqual(retargetedAgain.from)
  })

  test("renders a full-width pentagon with every vertex metric", () => {
    const profile = Option.getOrThrow(catalogRadarProfile(makeCatalogOnlyModel()))
    const frame = renderCatalogRadar(profile.values, profile.metrics)
    expect(frame).toHaveLength(15)
    const lines = frame.map((row) => row.map(({ text }) => text).join(""))
    expect(lines.every((line) => [...line].length === 56)).toBe(true)
    for (const { name, value } of profile.metrics) {
      expect(lines.join("\n")).toContain(name)
      expect(lines.join("\n")).toContain(value)
    }
  })

  test("never merges guide dots into profile-colored Braille cells", () => {
    expect(catalogRadarCell(0x01, 0x80)).toEqual({
      character: String.fromCodePoint(0x2801),
      tone: "profile",
    })
    expect(catalogRadarCell(0, 0x80)).toEqual({
      character: String.fromCodePoint(0x2880),
      tone: "grid",
    })
  })
})
