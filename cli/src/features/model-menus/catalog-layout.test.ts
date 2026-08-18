import { describe, expect, test } from "vitest"
import { getDisplayWidth } from "@magnitudedev/client-common"
import { ModelVariantLabelSchema } from "@magnitudedev/sdk"
import {
  CATALOG_INSPECTOR_WIDTH,
  CATALOG_INSPECTOR_CONTENT_WIDTH,
  CATALOG_SPLIT_INSPECTOR_HEIGHT,
  CATALOG_SPLIT_INSPECTOR_HEIGHTS,
  CATALOG_SPLIT_MIN_WIDTH,
  catalogDetailHints,
  catalogListHints,
  deriveCatalogLayout,
  formatCatalogModelLabel,
} from "./catalog-layout"
import { PENTAGON_RADAR_ROWS } from "../../components/pentagon-radar"

describe("catalog responsive layout", () => {
  test.each([
    [104, "list", false, true, true, true, false],
    [105, "split", true, false, false, false, false],
    [114, "split", true, false, false, false, false],
    [115, "split", true, true, false, false, false],
    [125, "split", true, true, false, false, false],
    [126, "split", true, true, true, false, false],
    [136, "split", true, true, true, false, false],
    [137, "split", true, true, true, true, false],
  ] as const)(
    "derives the %i-column layout",
    (width, mode, split, memory, speed, speculative, compact) => {
      const layout = deriveCatalogLayout(width)
      expect(layout.mode).toBe(mode)
      expect(layout.inspectorWidth > 0).toBe(split)
      expect(layout.showMemory).toBe(memory)
      expect(layout.showSpeed).toBe(speed)
      expect(layout.showSpeculative).toBe(speculative)
      expect(layout.compactHeader).toBe(compact)
      expect(layout.modelWidth).toBeGreaterThan(0)
    },
  )

  test("keeps the inspector fixed and gives the list the remainder", () => {
    for (const width of [CATALOG_SPLIT_MIN_WIDTH, 115, 137, 180]) {
      const layout = deriveCatalogLayout(width)
      expect(layout.inspectorWidth).toBe(CATALOG_INSPECTOR_WIDTH)
      expect(layout.listWidth + layout.dividerWidth + layout.inspectorWidth).toBe(width)
    }
  })

  test("preserves the established compact-header boundary", () => {
    expect(deriveCatalogLayout(81).compactHeader).toBe(true)
    expect(deriveCatalogLayout(82).compactHeader).toBe(false)
  })

  test("keeps every split inspector region inside the available content area", () => {
    expect(CATALOG_INSPECTOR_CONTENT_WIDTH).toBe(58)
    expect(CATALOG_SPLIT_INSPECTOR_HEIGHT).toBe(25)
    expect(CATALOG_SPLIT_INSPECTOR_HEIGHTS.metrics).toBe(PENTAGON_RADAR_ROWS)
    expect(CATALOG_SPLIT_INSPECTOR_HEIGHTS).toEqual({
      identity: 3,
      metrics: 15,
      info: 3,
      actions: 4,
    })
  })

  test("budgets every visible one-line column within the list content width", () => {
    for (const width of [40, 63, 74, 102, 103, 113, 124, 135, 180]) {
      const layout = deriveCatalogLayout(width)
      const visibleColumns = Object.values(layout.columns).filter((columnWidth) => columnWidth > 0)
      const allocated = 1
        + layout.modelWidth
        + visibleColumns.reduce((total, columnWidth) => total + columnWidth, 0)
        + layout.columnGap * visibleColumns.length
      expect(allocated).toBe(layout.contentWidth)
      expect(layout.columns.status).toBeGreaterThanOrEqual(getDisplayWidth("Update available"))
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

  test("preserves the established list and detail hints", () => {
    expect(catalogListHints(110)).toBe("↑↓ navigate · Enter details · D download · S select · Backspace cancel/remove · Esc close")
    expect(catalogListHints(95)).toBe("↑↓ move · Enter details · D download · S select · Esc close")
    expect(catalogListHints(81)).toBe("↑↓ move · Enter details · Esc close")
    expect(catalogDetailHints(true)).toBe("↑↓ choose · Enter select · Esc back")
    expect(catalogDetailHints(false)).toBe("↑↓ navigate · Enter choose · Esc back")
  })
})
