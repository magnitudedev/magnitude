import { Option } from "effect"
import { describe, expect, test } from "vitest"
import {
  generateRadarValues,
  interpolateRadarValues,
  radarCell,
  renderRadar,
  renderRadarCells,
  type RadarValues,
} from "./radar"

const values = (...measurements: number[]): RadarValues => measurements.map(Option.some)
const numbers = (profile: RadarValues): readonly number[] => profile.map(Option.getOrThrow)

describe("radar renderer", () => {
  test.each([3, 4, 5, 6, 8])("renders an exact frame for %i points", (pointCount) => {
    const frame = renderRadar(
      values(...Array.from({ length: pointCount }, (_, index) => (index + 1) / pointCount)),
      { columns: 20, rows: 10, guides: { ringCount: 3, spokes: true } },
    )
    expect(frame).toHaveLength(10)
    expect(frame.every((row) => row.map(({ text }) => text).join("").length === 20)).toBe(true)
    expect(frame.flat().some(({ tone }) => tone === "profile")).toBe(true)
    expect(frame.flat().some(({ tone }) => tone === "guide")).toBe(true)
  })

  test("supports disabled, rings-only, and spokes-only guides", () => {
    const profile = values(0.4, 0.6, 0.8, 0.7, 0.5)
    const disabled = renderRadarCells(profile, { columns: 16, rows: 8, guides: false })
    const rings = renderRadarCells(profile, {
      columns: 16,
      rows: 8,
      guides: { ringCount: 2, spokes: false },
    })
    const spokes = renderRadarCells(profile, {
      columns: 16,
      rows: 8,
      guides: { ringCount: 0, spokes: true },
    })
    expect(disabled.flat().some(({ tone }) => tone === "guide")).toBe(false)
    expect(rings.flat().some(({ tone }) => tone === "guide")).toBe(true)
    expect(spokes.flat().some(({ tone }) => tone === "guide")).toBe(true)
  })

  test("keeps missing measurements absent and never bridges their incident edges", () => {
    const complete = renderRadarCells(values(1, 1, 1, 1, 1), {
      columns: 20,
      rows: 10,
      guides: false,
    })
    const missing = renderRadarCells([
      Option.some(1),
      Option.none(),
      Option.some(1),
      Option.some(1),
      Option.some(1),
    ], {
      columns: 20,
      rows: 10,
      guides: false,
    })
    const profileCells = (frame: ReturnType<typeof renderRadarCells>) =>
      frame.flat().filter(({ tone }) => tone === "profile").length
    expect(profileCells(missing)).toBeLessThan(profileCells(complete))
    expect(profileCells(missing)).toBeGreaterThan(0)
  })

  test("interpolates arbitrary equal profiles and follows target presence", () => {
    expect(numbers(interpolateRadarValues(values(0, 0, 0), values(1, 0.5, 0.25), 0.5)))
      .toEqual([0.5, 0.25, 0.125])
    expect(interpolateRadarValues(
      [Option.some(1), Option.none(), Option.some(0)],
      [Option.none(), Option.some(0.5), Option.some(1)],
      0.5,
    )).toEqual([Option.none(), Option.some(0.5), Option.some(0.5)])
    expect(() => interpolateRadarValues(values(0, 0, 0), values(1, 1, 1, 1), 0.5))
      .toThrow("equal profile lengths")
  })

  test("generates stable values of the requested size and range", () => {
    const first = generateRadarValues({
      seed: 42,
      targetIndex: 3,
      pointCount: 6,
      valueRange: [0.25, 0.75],
    })
    const repeated = generateRadarValues({
      seed: 42,
      targetIndex: 3,
      pointCount: 6,
      valueRange: [0.25, 0.75],
    })
    const next = generateRadarValues({
      seed: 42,
      targetIndex: 4,
      pointCount: 6,
      valueRange: [0.25, 0.75],
    })
    expect(first).toEqual(repeated)
    expect(first).not.toEqual(next)
    expect(numbers(first)).toHaveLength(6)
    expect(numbers(first).every((value) => value >= 0.25 && value <= 0.75)).toBe(true)
  })

  test("gives profile dots precedence over guide dots", () => {
    expect(radarCell(0x01, 0x80)).toEqual({
      character: String.fromCodePoint(0x2801),
      tone: "profile",
    })
    expect(radarCell(0, 0x80)).toEqual({
      character: String.fromCodePoint(0x2880),
      tone: "guide",
    })
  })

  test("maps all eight Braille dots", () => {
    for (const mask of [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80]) {
      expect(radarCell(mask, 0)).toEqual({
        character: String.fromCodePoint(0x2800 + mask),
        tone: "profile",
      })
    }
  })
})
