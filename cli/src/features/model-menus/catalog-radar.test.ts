import { describe, expect, test } from "vitest"
import { Option } from "effect"
import {
  PENTAGON_RADAR_DURATION_MS,
  interpolatePentagonRadar,
  pentagonRadarEase,
  pentagonRadarTransitionValues,
  renderPentagonRadar,
  retargetPentagonRadar,
  type PentagonRadarValues,
} from "../../components/pentagon-radar"
import { radarCell } from "../../components/radar"
import { localModelRadarAxes } from "@magnitudedev/client-common"
import {
  GIB,
  makeCatalogOnlyModel,
  TEST_MEMORY_DOMAIN_ID,
} from "../local-inference/test-fixtures"

const zero: PentagonRadarValues = [0, 0, 0, 0, 0].map(Option.some) as unknown as PentagonRadarValues
const one: PentagonRadarValues = [1, 1, 1, 1, 1].map(Option.some) as unknown as PentagonRadarValues

describe("catalog radar", () => {
  test("derives all five catalog axes and their displayed metrics", () => {
    const axes = Option.getOrThrow(localModelRadarAxes(makeCatalogOnlyModel()))
    expect(Option.getOrThrow(axes[0].value)).toBe(0.75)
    expect(Option.getOrThrow(axes[1].value)).toBeGreaterThanOrEqual(0)
    expect(Option.getOrThrow(axes[2].value)).toBe(0)
    expect(Option.getOrThrow(axes[3].value)).toBe(0)
    expect(Option.getOrThrow(axes[4].value)).toBe(0.75)
    expect(axes.map(({ label }) => label)).toEqual([
      "INTELLIGENCE",
      "SPEED",
      "SPECULATION",
      "MEMORY",
      "ACCURACY",
    ])
  })

  test("omits the profile when assessment has no performance samples", () => {
    const model = makeCatalogOnlyModel()
    if (
      model.servingState._tag !== "Assessed" ||
      model.servingState.assessment._tag !== "Fits" ||
      !("rankingScores" in model.servingState)
    ) {
      throw new Error("Expected an assessed catalog fixture")
    }
    const servingState = model.servingState
    const assessment = servingState.assessment
    const withoutPerformance = {
      ...model,
      servingState: {
        ...servingState,
        assessment: {
          ...assessment,
          performance: [],
        },
      },
    }
    expect(Option.isNone(localModelRadarAxes(withoutPerformance))).toBe(true)
  })

  test("displays memory using the hardware convention", () => {
    const model = makeCatalogOnlyModel()
    if (model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits"
      || !("rankingScores" in model.servingState)) {
      throw new Error("Catalog radar fixture must have a fitting assessment")
    }
    const servingState = model.servingState
    const assessed = servingState.assessment
    const axes = Option.getOrThrow(localModelRadarAxes({
      ...model,
      servingState: {
        ...servingState,
        assessment: {
          ...assessed,
          memory: { ...assessed.memory, totalRequiredBytes: 3.4 * 1024 ** 3 },
        },
      },
    }))

    expect(axes[3].detail).toBe("Tiny (3.4 GB)")
  })

  test("plots larger memory footprints farther from the center", () => {
    const model = makeCatalogOnlyModel()
    if (model.servingState._tag !== "Assessed"
      || model.servingState.assessment._tag !== "Fits"
      || !("rankingScores" in model.servingState)) {
      throw new Error("Catalog radar fixture must have a fitting assessment")
    }
    const modelServingState = model.servingState
    const assessed = modelServingState.assessment
    const servingState = { ...modelServingState, assessment: assessed }
    const memoryValue = (requiredBytes: number) => {
      const axes = Option.getOrThrow(localModelRadarAxes({
        ...model,
        servingState: {
          ...servingState,
          assessment: {
            ...assessed,
            memory: {
              ...assessed.memory,
              domains: [{
                memoryDomainId: TEST_MEMORY_DOMAIN_ID,
                capacityBytes: 12 * GIB,
                compatibilityReserveBytes: 2 * GIB,
                remainingBytes: 10 * GIB - requiredBytes,
                requiredBytes,
              }],
              totalRequiredBytes: requiredBytes,
            },
          },
        },
      }))
      return Option.getOrThrow(axes[3].value)
    }

    expect(memoryValue(2 * GIB)).toBeCloseTo(0.2)
    expect(memoryValue(8 * GIB)).toBeCloseTo(0.8)
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

  test("retains the chart's supported minimum frame", () => {
    const axes = Option.getOrThrow(localModelRadarAxes(makeCatalogOnlyModel()))
    const frame = renderPentagonRadar(axes, undefined, 44, 13)
    expect(frame).toHaveLength(13)
    expect(frame.every((row) => row.map(({ text }) => text).join("").length === 44)).toBe(true)
  })

  test("never merges guide dots into profile-colored Braille cells", () => {
    expect(radarCell(0x01, 0x80)).toEqual({
      character: String.fromCodePoint(0x2801),
      tone: "profile",
    })
    expect(radarCell(0, 0x80)).toEqual({
      character: String.fromCodePoint(0x2880),
      tone: "guide",
    })
  })
})
