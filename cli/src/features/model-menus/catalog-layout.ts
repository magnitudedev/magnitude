import {
  getDisplayWidth,
  truncateToDisplayWidth,
} from "@magnitudedev/client-common"
import type { ModelVariantLabel } from "@magnitudedev/sdk"

export type CatalogLayoutMode = "full" | "quality" | "compact" | "stacked" | "minimal"

export interface CatalogColumnWidths {
  readonly recommendation: number
  readonly memory: number
  readonly intelligence: number
  readonly quality: number
  readonly speed: number
  readonly status: number
}

export interface CatalogLayout {
  readonly mode: CatalogLayoutMode
  readonly contentWidth: number
  readonly modelWidth: number
  readonly columns: CatalogColumnWidths
  readonly showIntelligence: boolean
  readonly showQuality: boolean
  readonly showSpeed: boolean
  readonly stackedRows: boolean
  readonly compactHeader: boolean
}

const HORIZONTAL_PADDING = 4
const CURSOR_WIDTH = 2

const FULL_COLUMNS: CatalogColumnWidths = {
  recommendation: 15,
  memory: 10,
  intelligence: 13,
  quality: 13,
  speed: 12,
  status: 17,
}

const COMPACT_COLUMNS: CatalogColumnWidths = {
  recommendation: 15,
  memory: 10,
  intelligence: 0,
  quality: 0,
  speed: 12,
  status: 17,
}

const QUALITY_COLUMNS: CatalogColumnWidths = {
  ...COMPACT_COLUMNS,
  quality: 13,
}

const EMPTY_COLUMNS: CatalogColumnWidths = {
  recommendation: 0,
  memory: 0,
  intelligence: 0,
  quality: 0,
  speed: 0,
  status: 0,
}

const columnTotal = (columns: CatalogColumnWidths): number =>
  columns.recommendation
  + columns.memory
  + columns.intelligence
  + columns.quality
  + columns.speed
  + columns.status

export const deriveCatalogLayout = (measuredWidth: number): CatalogLayout => {
  const width = Math.max(1, Math.floor(measuredWidth))
  const contentWidth = Math.max(1, width - HORIZONTAL_PADDING)

  if (width >= 110) {
    return {
      mode: "full",
      contentWidth,
      modelWidth: Math.max(1, contentWidth - CURSOR_WIDTH - columnTotal(FULL_COLUMNS)),
      columns: FULL_COLUMNS,
      showIntelligence: true,
      showQuality: true,
      showSpeed: true,
      stackedRows: false,
      compactHeader: false,
    }
  }

  if (width >= 95) {
    return {
      mode: "quality",
      contentWidth,
      modelWidth: Math.max(1, contentWidth - CURSOR_WIDTH - columnTotal(QUALITY_COLUMNS)),
      columns: QUALITY_COLUMNS,
      showIntelligence: false,
      showQuality: true,
      showSpeed: true,
      stackedRows: false,
      compactHeader: false,
    }
  }

  if (width >= 82) {
    return {
      mode: "compact",
      contentWidth,
      modelWidth: Math.max(1, contentWidth - CURSOR_WIDTH - columnTotal(COMPACT_COLUMNS)),
      columns: COMPACT_COLUMNS,
      showIntelligence: false,
      showQuality: false,
      showSpeed: true,
      stackedRows: false,
      compactHeader: false,
    }
  }

  const minimal = width < 56
  return {
    mode: minimal ? "minimal" : "stacked",
    contentWidth,
    modelWidth: Math.max(1, contentWidth - CURSOR_WIDTH),
    columns: EMPTY_COLUMNS,
    showIntelligence: false,
    showQuality: false,
    showSpeed: !minimal,
    stackedRows: true,
    compactHeader: true,
  }
}

export const formatCatalogModelLabel = (
  displayName: string,
  variantLabel: ModelVariantLabel,
  maxWidth: number,
): string => {
  const safeWidth = Math.max(1, Math.floor(maxWidth))
  const suffix = ` (${variantLabel})`
  const suffixWidth = getDisplayWidth(suffix)
  const fullLabel = `${displayName}${suffix}`

  if (getDisplayWidth(fullLabel) <= safeWidth || suffixWidth >= safeWidth) {
    return truncateToDisplayWidth(fullLabel, safeWidth)
  }

  return `${truncateToDisplayWidth(displayName, safeWidth - suffixWidth)}${suffix}`
}

export const catalogListHints = (mode: CatalogLayoutMode): string => {
  if (mode === "full") {
    return "↑↓ navigate · Enter details · D download · S select · Backspace cancel/remove · Esc close"
  }
  if (mode === "quality" || mode === "compact") {
    return "↑↓ move · Enter details · D download · S select · Esc close"
  }
  return "↑↓ move · Enter details · Esc close"
}

export const catalogDetailHints = (compactHeader: boolean): string =>
  compactHeader
    ? "↑↓ choose · Enter select · Esc back"
    : "↑↓ navigate · Enter choose · Esc back"
