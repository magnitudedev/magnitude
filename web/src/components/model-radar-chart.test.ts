import { describe, expect, test } from "vitest"
import { radarPolygonPoints } from "./model-radar-chart"

describe("model radar chart", () => {
  test("creates one finite point for every axis", () => {
    const points = radarPolygonPoints([1, 0.75, 0.5, 0.25, 0])
    expect(points.split(" ")).toHaveLength(5)
    expect(points).not.toContain("NaN")
    expect(points).not.toContain("Infinity")
  })

  test("applies grid scale without changing axis count", () => {
    expect(radarPolygonPoints([1, 1, 1, 1, 1], 0.5).split(" ")).toHaveLength(5)
  })
})
