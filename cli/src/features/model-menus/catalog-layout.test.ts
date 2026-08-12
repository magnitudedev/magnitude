import { describe, expect, test } from "vitest"
import { getDisplayWidth } from "@magnitudedev/client-common"
import { ModelVariantLabelSchema } from "@magnitudedev/sdk"
import {
  catalogDetailHints,
  catalogListHints,
  deriveCatalogLayout,
  formatCatalogModelLabel,
} from "./catalog-layout"

describe("catalog responsive layout", () => {
  test.each([
    [160, "full", true, true, true, false],
    [110, "full", true, true, true, false],
    [109, "quality", false, true, true, false],
    [95, "quality", false, true, true, false],
    [94, "compact", false, false, true, false],
    [82, "compact", false, false, true, false],
    [81, "stacked", false, false, true, true],
    [56, "stacked", false, false, true, true],
    [55, "minimal", false, false, false, true],
    [40, "minimal", false, false, false, true],
  ] as const)(
    "derives the %i-column layout",
    (width, mode, intelligence, quality, speed, stacked) => {
      const layout = deriveCatalogLayout(width)

      expect(layout.mode).toBe(mode)
      expect(layout.showIntelligence).toBe(intelligence)
      expect(layout.showQuality).toBe(quality)
      expect(layout.showSpeed).toBe(speed)
      expect(layout.stackedRows).toBe(stacked)
      expect(layout.modelWidth).toBeGreaterThan(0)
    },
  )

  test("budgets every full and compact column within the content width", () => {
    for (const width of [82, 90, 94, 95, 100, 109, 110, 120, 160]) {
      const layout = deriveCatalogLayout(width)
      const columns = layout.columns
      const allocated = 2
        + layout.modelWidth
        + columns.recommendation
        + columns.memory
        + columns.intelligence
        + columns.quality
        + columns.speed
        + columns.status

      expect(allocated).toBe(layout.contentWidth)
    }
  })

  test("truncates the model name while preserving the variant label when it fits", () => {
    const label = formatCatalogModelLabel(
      "A Very Long Coding Model Name",
      ModelVariantLabelSchema.make("Q4 QAT"),
      20,
    )

    expect(label).toContain("…")
    expect(label.endsWith(" (Q4 QAT)")).toBe(true)
    expect(getDisplayWidth(label)).toBeLessThanOrEqual(20)
  })

  test("never exceeds tiny display-width budgets", () => {
    for (const width of [1, 4, 8, 12]) {
      const label = formatCatalogModelLabel(
        "模型🚀 Coding Model",
        ModelVariantLabelSchema.make("NVFP4"),
        width,
      )
      expect(getDisplayWidth(label)).toBeLessThanOrEqual(width)
    }
  })

  test("shortens help copy as comparison evidence collapses", () => {
    expect(catalogListHints("full")).toContain("Backspace")
    expect(catalogListHints("quality")).not.toContain("Backspace")
    expect(catalogListHints("compact")).not.toContain("Backspace")
    expect(catalogListHints("stacked")).toBe("↑↓ move · Enter details · Esc close")
    expect(catalogDetailHints(true)).toBe("↑↓ choose · Enter select · Esc back")
  })
})
