import {
  getDisplayWidth,
  truncateToDisplayWidth,
} from "@magnitudedev/client-common"
import type { ModelVariantLabel } from "@magnitudedev/sdk"

export type CatalogLayoutMode = "list" | "split"

export interface CatalogColumnWidths {
  readonly memory: number
  readonly speed: number
  readonly speculative: number
  readonly status: number
}

export interface CatalogLayout {
  readonly mode: CatalogLayoutMode
  readonly measuredWidth: number
  readonly listWidth: number
  readonly inspectorWidth: number
  readonly dividerWidth: number
  readonly contentWidth: number
  readonly modelWidth: number
  readonly columnGap: number
  readonly columns: CatalogColumnWidths
  readonly showMemory: boolean
  readonly showSpeed: boolean
  readonly showSpeculative: boolean
  readonly compactHeader: boolean
}

const HORIZONTAL_PADDING = 4
export const CATALOG_INSPECTOR_WIDTH = 60
export const CATALOG_INSPECTOR_CONTENT_WIDTH = CATALOG_INSPECTOR_WIDTH - HORIZONTAL_PADDING
export const CATALOG_DIVIDER_WIDTH = 1
export const CATALOG_MIN_LIST_WIDTH = 42
export const CATALOG_SPLIT_MIN_WIDTH = CATALOG_INSPECTOR_WIDTH
  + CATALOG_DIVIDER_WIDTH
  + CATALOG_MIN_LIST_WIDTH
export const CATALOG_SPLIT_INSPECTOR_HEIGHTS = {
  identity: 4,
  metrics: 9,
  info: 3,
  recommendation: 5,
  actions: 4,
} as const
export const CATALOG_SPLIT_INSPECTOR_HEIGHT = Object.values(
  CATALOG_SPLIT_INSPECTOR_HEIGHTS,
).reduce((total, height) => total + height, 0)

const CURSOR_WIDTH = 1
const COLUMN_GAP = 1
const STATUS_WIDTH = 16
const MEMORY_WIDTH = 9
const SPEED_WIDTH = 10
const SPECULATIVE_WIDTH = 12

const columnsForListWidth = (listWidth: number): CatalogColumnWidths => ({
  memory: listWidth >= 52 ? MEMORY_WIDTH : 0,
  speed: listWidth >= 63 ? SPEED_WIDTH : 0,
  speculative: listWidth >= 74 ? SPECULATIVE_WIDTH : 0,
  status: STATUS_WIDTH,
})

export const deriveCatalogLayout = (measuredWidth: number): CatalogLayout => {
  const width = Math.max(1, Math.floor(measuredWidth))
  const split = width >= CATALOG_SPLIT_MIN_WIDTH
  const inspectorWidth = split ? CATALOG_INSPECTOR_WIDTH : 0
  const dividerWidth = split ? CATALOG_DIVIDER_WIDTH : 0
  const listWidth = Math.max(1, width - inspectorWidth - dividerWidth)
  const contentWidth = Math.max(1, listWidth - HORIZONTAL_PADDING)
  const columns = columnsForListWidth(listWidth)
  const visibleColumns = Object.values(columns).filter((columnWidth) => columnWidth > 0).length
  const fixedWidth = CURSOR_WIDTH
    + Object.values(columns).reduce((total, columnWidth) => total + columnWidth, 0)
    + COLUMN_GAP * visibleColumns

  return {
    mode: split ? "split" : "list",
    measuredWidth: width,
    listWidth,
    inspectorWidth,
    dividerWidth,
    contentWidth,
    modelWidth: Math.max(1, contentWidth - fixedWidth),
    columnGap: COLUMN_GAP,
    columns,
    showMemory: columns.memory > 0,
    showSpeed: columns.speed > 0,
    showSpeculative: columns.speculative > 0,
    compactHeader: width < 82,
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

export const catalogListHints = (measuredWidth: number): string => {
  if (measuredWidth >= 110) {
    return "↑↓ navigate · Enter details · D download · S select · Backspace cancel/remove · Esc close"
  }
  if (measuredWidth >= 82) {
    return "↑↓ move · Enter details · D download · S select · Esc close"
  }
  return "↑↓ move · Enter details · Esc close"
}

export const catalogDetailHints = (compactHeader: boolean): string =>
  compactHeader
    ? "↑↓ choose · Enter select · Esc back"
    : "↑↓ navigate · Enter choose · Esc back"
