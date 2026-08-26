import { Option } from "effect"
import { describe, expect, test } from "vitest"
import { generateRadarValues } from "./radar"
import {
  radarLoaderEase,
  radarLoaderPulseIntensity,
  radarLoaderValues,
} from "./radar-loader"

const numbers = (profile: ReturnType<typeof radarLoaderValues>): readonly number[] =>
  profile.map(Option.getOrThrow)

describe("radar loader motion", () => {
  test("uses a bounded cosine ease-in-out", () => {
    expect(radarLoaderEase(-1)).toBe(0)
    expect(radarLoaderEase(0)).toBe(0)
    expect(radarLoaderEase(0.5)).toBeCloseTo(0.5)
    expect(radarLoaderEase(1)).toBe(1)
    expect(radarLoaderEase(2)).toBe(1)
  })

  test("propagates a localized pulse clockwise over twice the movement period", () => {
    const intensity = (timeMs: number, position: number) => radarLoaderPulseIntensity({
      timeMs,
      epochMs: 0,
      movementPeriodMs: 800,
      position,
    })
    expect(intensity(0, 0)).toBeCloseTo(1)
    expect(intensity(400, 0.25)).toBeCloseTo(1)
    expect(intensity(800, 0.5)).toBeCloseTo(1)
    expect(intensity(1_200, 0.75)).toBeCloseTo(1)
    expect(intensity(1_600, 0)).toBeCloseTo(1)
    expect(intensity(0, 0.5)).toBe(0)
  })

  test("uses a continuous nonlinear gradient and wraps around the profile", () => {
    const intensity = (position: number) => radarLoaderPulseIntensity({
      timeMs: 0,
      epochMs: 0,
      movementPeriodMs: 800,
      position,
    })
    expect(intensity(0)).toBeCloseTo(1)
    expect(intensity(0.25)).toBeCloseTo(0.5)
    expect(intensity(0.5)).toBeCloseTo(0)
    expect(intensity(0.75)).toBeCloseTo(0.5)
    expect(intensity(0.94)).toBeCloseTo(intensity(0.06))
  })

  test("starts at a generated profile and reaches the next target at the interval boundary", () => {
    const input = {
      epochMs: 100,
      seed: 77,
      pointCount: 5,
      intervalMs: 800,
      transitionMs: 800,
      valueRange: [0.25, 1] as const,
    }
    expect(radarLoaderValues({ ...input, timeMs: 100 })).toEqual(generateRadarValues({
      seed: 77,
      targetIndex: 0,
      pointCount: 5,
      valueRange: [0.25, 1],
    }))
    expect(radarLoaderValues({ ...input, timeMs: 900 })).toEqual(generateRadarValues({
      seed: 77,
      targetIndex: 1,
      pointCount: 5,
      valueRange: [0.25, 1],
    }))
  })

  test("holds the target after a shorter transition", () => {
    const input = {
      epochMs: 0,
      seed: 19,
      pointCount: 4,
      intervalMs: 800,
      transitionMs: 200,
      valueRange: [0.3, 0.9] as const,
    }
    expect(radarLoaderValues({ ...input, timeMs: 200 }))
      .toEqual(radarLoaderValues({ ...input, timeMs: 700 }))
  })

  test("is stable without time advancement and remains in range", () => {
    const input = {
      timeMs: 475,
      epochMs: 25,
      seed: 5,
      pointCount: 7,
      intervalMs: 800,
      transitionMs: 800,
      valueRange: [0.4, 0.6] as const,
    }
    const first = radarLoaderValues(input)
    expect(first).toEqual(radarLoaderValues(input))
    expect(numbers(first)).toHaveLength(7)
    expect(numbers(first).every((value) => value >= 0.4 && value <= 0.6)).toBe(true)
  })
})
