import { describe, expect, test } from "vitest"
import { Option } from "effect"
import {
  CATALOG_RADAR_DURATION_MS,
  catalogRadarCell,
  catalogRadarEase,
  catalogRadarProfile,
  catalogRadarTransitionValues,
  interpolateCatalogRadar,
  normalizeCatalogRadarSpeed,
  renderCatalogRadar,
  retargetCatalogRadar,
  type CatalogRadarValues,
} from "./catalog-radar"
import { makeCatalogOnlyModel } from "../local-inference/test-fixtures"

const zero: CatalogRadarValues = [0, 0, 0]
const one: CatalogRadarValues = [1, 1, 1]

describe("catalog radar", () => {
  test("normalizes speed logarithmically between the policy floor and ceiling", () => {
    expect(normalizeCatalogRadarSpeed(1)).toBe(0)
    expect(normalizeCatalogRadarSpeed(5)).toBe(0)
    expect(normalizeCatalogRadarSpeed(60)).toBeCloseTo(1)
    expect(normalizeCatalogRadarSpeed(600)).toBe(1)
    expect(normalizeCatalogRadarSpeed(Math.sqrt(5 * 60))).toBeCloseTo(0.5)
  })

  test("derives intelligence, speed, and accuracy without requiring memory evidence", () => {
    const profile = Option.getOrThrow(catalogRadarProfile(makeCatalogOnlyModel()))
    expect(profile.values[0]).toBe(0.75)
    expect(profile.values[1]).toBeGreaterThanOrEqual(0)
    expect(profile.values[2]).toBe(0.75)
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
    const third: CatalogRadarValues = [0.2, 0.6, 1]
    const retargeted = retargetCatalogRadar(one, third, first, 100 + CATALOG_RADAR_DURATION_MS / 2)
    expect(retargeted.from).toEqual(halfway)
    const fourth: CatalogRadarValues = [1, 0.6, 0.2]
    const retargetedAgain = retargetCatalogRadar(third, fourth, retargeted, 180)
    expect(retargetedAgain.from).toEqual(catalogRadarTransitionValues(retargeted, 180))
    expect(catalogRadarTransitionValues(retargetedAgain, 180)).toEqual(retargetedAgain.from)
  })

  test("renders a compact fixed-size triangle with all three labels", () => {
    const frame = renderCatalogRadar([0.8, 0.55, 0.9])
    expect(frame).toHaveLength(9)
    const lines = frame.map((row) => row.map(({ text }) => text).join(""))
    expect(lines.every((line) => [...line].length === 26)).toBe(true)
    expect(lines.join("\n")).toContain("INT")
    expect(lines.join("\n")).toContain("ACC")
    expect(lines.join("\n")).toContain("SPD")
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
