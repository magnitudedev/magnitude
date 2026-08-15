import { Option } from "effect"
import { localModelSpeculativeMethodLabel } from "@magnitudedev/client-common"
import type { LocalModel } from "@magnitudedev/sdk"
import {
  performanceRangeSpeedLabel,
} from "../local-inference/view-model"

export const CATALOG_RADAR_DURATION_MS = 220
export const CATALOG_RADAR_COLUMNS = 56
export const CATALOG_RADAR_ROWS = 15

// Axis order follows the pentagon clockwise from its top vertex.
export type CatalogRadarValues = readonly [number, number, number, number, number]

export interface CatalogRadarMetric {
  readonly name: string
  readonly value: string
}

export type CatalogRadarMetrics = readonly [
  CatalogRadarMetric,
  CatalogRadarMetric,
  CatalogRadarMetric,
  CatalogRadarMetric,
  CatalogRadarMetric,
]

export interface CatalogRadarProfile {
  readonly values: CatalogRadarValues
  readonly metrics: CatalogRadarMetrics
}

export interface CatalogRadarTransition {
  readonly from: CatalogRadarValues
  readonly to: CatalogRadarValues
  readonly startedAt: number
}

export type CatalogRadarTone = "empty" | "grid" | "profile" | "label" | "detail"

export interface CatalogRadarRun {
  readonly text: string
  readonly tone: CatalogRadarTone
}

export type CatalogRadarFrame = readonly (readonly CatalogRadarRun[])[]

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value))

export const normalizeCatalogRadarSpeed = (tokensPerSecond: number): number => {
  if (tokensPerSecond <= 30) return clamp01(tokensPerSecond / 30) * 0.5
  if (tokensPerSecond <= 100) return 0.5 + ((tokensPerSecond - 30) / 70) * 0.4
  if (tokensPerSecond >= 1_000) return 1
  return 0.9 + 0.1 * Math.log(tokensPerSecond / 100) / Math.log(10)
}

const GIB = 1024 ** 3

const compactMemorySize = (bytes: number): string => {
  const gigabytes = bytes / GIB
  return `${gigabytes.toFixed(gigabytes >= 10 ? 0 : 1)} GB`
}

const catalogRadarMemoryUseRatio = (
  assessment: Extract<LocalModel["servingState"], { readonly _tag: "Assessed" }>["assessment"],
): number => {
  if (assessment._tag !== "Fits") return 1
  return assessment.memory.domains.reduce((highestUse, domain) => {
    const usableCapacityBytes = domain.capacityBytes - domain.compatibilityReserveBytes
    const use = usableCapacityBytes <= 0
      ? (domain.requiredBytes > 0 ? 1 : 0)
      : domain.requiredBytes / usableCapacityBytes
    return Math.max(highestUse, clamp01(use))
  }, 0)
}

export const normalizeCatalogRadarMemoryEfficiency = (
  assessment: Extract<LocalModel["servingState"], { readonly _tag: "Assessed" }>["assessment"],
): number => 1 - catalogRadarMemoryUseRatio(assessment)

const catalogRadarSpeculationValue = (model: LocalModel): number => {
  if (model.bundle._tag !== "SpeculativeDecoding") return 0
  switch (model.bundle.method._tag) {
    case "Mtp": return 1 / 3
    case "DFlash": return 2 / 3
    case "DSpark": return 1
  }
}

const accuracyLabel = (model: LocalModel): string => {
  if (model.catalogMembershipState._tag !== "InCatalog") return "—"
  const rank = model.catalogMembershipState.catalogData.fidelityRank
  return rank >= 75 ? "Native"
    : rank >= 55 ? "Very high"
      : rank >= 45 ? "High" : "Reduced"
}

const shortVariantLabel = (model: LocalModel): string =>
  String(model.presentation.variantLabel).match(/\b(?:IQ|Q)\d+(?:\.\d+)?\b/i)?.[0]
    ?? String(model.presentation.variantLabel).split(/[ _-]/, 1)[0]
    ?? String(model.presentation.variantLabel)

const memoryFootprintLabel = (
  assessment: Extract<LocalModel["servingState"], { readonly _tag: "Assessed" }>["assessment"],
): string => {
  const use = catalogRadarMemoryUseRatio(assessment)
  if (use <= 0.2) return "Light"
  if (use <= 0.4) return "Moderate"
  if (use <= 0.6) return "Substantial"
  if (use <= 0.8) return "Heavy"
  return "Near capacity"
}

export const catalogRadarProfile = (model: LocalModel): Option.Option<CatalogRadarProfile> => {
  if (model.catalogMembershipState._tag !== "InCatalog"
    || model.servingState._tag !== "Assessed"
    || model.servingState.assessment._tag !== "Fits") return Option.none()

  const catalog = model.catalogMembershipState.catalogData
  const assessment = model.servingState.assessment
  const comparisonContext = Math.min(50_000, assessment.profile.contextLength)
  const performance = assessment.performance.find(({ contextTokens }) =>
    contextTokens === comparisonContext)
  if (performance === undefined) return Option.none()

  const values: CatalogRadarValues = [
    clamp01(catalog.intelligenceScore / 100),
    normalizeCatalogRadarSpeed(performance.estimatedTokensPerSecond),
    catalogRadarSpeculationValue(model),
    normalizeCatalogRadarMemoryEfficiency(assessment),
    clamp01(catalog.fidelityRank / 100),
  ]
  if (values.some((value) => !Number.isFinite(value))) return Option.none()

  const speculation = Option.getOrElse(localModelSpeculativeMethodLabel(model), () => "None")

  return Option.some({
    values,
    metrics: [
      { name: "INTELLIGENCE", value: `${Math.round(catalog.intelligenceScore)}%` },
      { name: "SPEED", value: performanceRangeSpeedLabel(model, "tok/s") },
      { name: "SPECULATION", value: speculation },
      {
        name: "MEMORY",
        value: `${memoryFootprintLabel(assessment)} (${compactMemorySize(assessment.memory.totalRequiredBytes)})`,
      },
      {
        name: "ACCURACY",
        value: `${accuracyLabel(model)} (${shortVariantLabel(model)})`,
      },
    ],
  })
}

export const catalogRadarEase = (progress: number): number =>
  1 - (1 - clamp01(progress)) ** 3

export const interpolateCatalogRadar = (
  from: CatalogRadarValues,
  to: CatalogRadarValues,
  progress: number,
): CatalogRadarValues => {
  const eased = catalogRadarEase(progress)
  return from.map((value, index) => value + (to[index]! - value) * eased) as unknown as CatalogRadarValues
}

export const catalogRadarTransitionValues = (
  transition: CatalogRadarTransition,
  now: number,
): CatalogRadarValues => interpolateCatalogRadar(
  transition.from,
  transition.to,
  (now - transition.startedAt) / CATALOG_RADAR_DURATION_MS,
)

export const retargetCatalogRadar = (
  current: CatalogRadarValues,
  next: CatalogRadarValues,
  previous: CatalogRadarTransition | null,
  now: number,
): CatalogRadarTransition => ({
  from: previous === null ? current : catalogRadarTransitionValues(previous, now),
  to: next,
  startedAt: now,
})

interface Point {
  readonly x: number
  readonly y: number
}

interface Radii {
  readonly x: number
  readonly y: number
}

const BRAILLE_BITS = [
  [0x01, 0x08],
  [0x02, 0x10],
  [0x04, 0x20],
  [0x40, 0x80],
] as const

const pointOnAxis = (center: Point, radii: Radii, axis: number): Point => {
  const angle = -Math.PI / 2 + axis * Math.PI * 2 / 5
  return {
    x: center.x + Math.cos(angle) * radii.x,
    y: center.y + Math.sin(angle) * radii.y,
  }
}

const setDot = (buffer: Uint8Array, dotWidth: number, dotHeight: number, x: number, y: number): void => {
  const roundedX = Math.round(x)
  const roundedY = Math.round(y)
  if (roundedX < 0 || roundedY < 0 || roundedX >= dotWidth || roundedY >= dotHeight) return
  buffer[roundedY * dotWidth + roundedX] = 1
}

const drawLine = (
  buffer: Uint8Array,
  dotWidth: number,
  dotHeight: number,
  start: Point,
  end: Point,
): void => {
  let x0 = Math.round(start.x)
  let y0 = Math.round(start.y)
  const x1 = Math.round(end.x)
  const y1 = Math.round(end.y)
  const dx = Math.abs(x1 - x0)
  const sx = x0 < x1 ? 1 : -1
  const dy = -Math.abs(y1 - y0)
  const sy = y0 < y1 ? 1 : -1
  let error = dx + dy
  for (;;) {
    setDot(buffer, dotWidth, dotHeight, x0, y0)
    if (x0 === x1 && y0 === y1) break
    const doubled = error * 2
    if (doubled >= dy) {
      error += dy
      x0 += sx
    }
    if (doubled <= dx) {
      error += dx
      y0 += sy
    }
  }
}

const brailleMask = (
  buffer: Uint8Array,
  dotWidth: number,
  cellX: number,
  cellY: number,
): number => {
  let mask = 0
  for (let y = 0; y < 4; y += 1) {
    for (let x = 0; x < 2; x += 1) {
      if (buffer[(cellY * 4 + y) * dotWidth + cellX * 2 + x] === 1) {
        mask |= BRAILLE_BITS[y]![x]!
      }
    }
  }
  return mask
}

const writeLabel = (
  characters: string[][],
  tones: CatalogRadarTone[][],
  text: string,
  startX: number,
  y: number,
  tone: Extract<CatalogRadarTone, "label" | "detail">,
): void => {
  if (y < 0 || y >= characters.length) return
  for (let index = 0; index < text.length; index += 1) {
    const x = startX + index
    if (x < 0 || x >= characters[y]!.length) continue
    characters[y]![x] = text[index]!
    tones[y]![x] = tone
  }
}

type MetricAlignment = "left" | "center" | "right"

const metricStartX = (
  text: string,
  anchorX: number,
  alignment: MetricAlignment,
): number => alignment === "left"
  ? anchorX
  : alignment === "right"
    ? anchorX - text.length + 1
    : Math.round(anchorX - text.length / 2)

const writeMetric = (
  characters: string[][],
  tones: CatalogRadarTone[][],
  metric: CatalogRadarMetric,
  anchorX: number,
  y: number,
  alignment: MetricAlignment,
): void => {
  writeLabel(
    characters,
    tones,
    metric.name,
    metricStartX(metric.name, anchorX, alignment),
    y,
    "label",
  )
  writeLabel(
    characters,
    tones,
    metric.value,
    metricStartX(metric.value, anchorX, alignment),
    y + 1,
    "detail",
  )
}

export const renderCatalogRadar = (
  values: CatalogRadarValues,
  metrics: CatalogRadarMetrics,
  columns = CATALOG_RADAR_COLUMNS,
  rows = CATALOG_RADAR_ROWS,
): CatalogRadarFrame => {
  const safeColumns = Math.max(44, Math.floor(columns))
  const safeRows = Math.max(13, Math.floor(rows))
  const dotWidth = safeColumns * 2
  const dotHeight = safeRows * 4
  const chartTop = 8
  const radius = 24
  const upperLabelGap = 2
  const lowerLabelGap = 3
  const radii = { x: radius, y: radius }
  const center = {
    x: dotWidth / 2,
    y: chartTop + radius,
  }
  const grid = new Uint8Array(dotWidth * dotHeight)
  const profile = new Uint8Array(dotWidth * dotHeight)

  for (let axis = 0; axis < 5; axis += 1) {
    drawLine(grid, dotWidth, dotHeight, center, pointOnAxis(center, radii, axis))
  }
  for (const scale of [1 / 3, 2 / 3, 1]) {
    const scaledRadii = { x: radii.x * scale, y: radii.y * scale }
    const points = values.map((_, axis) => pointOnAxis(center, scaledRadii, axis))
    for (let axis = 0; axis < 5; axis += 1) {
      drawLine(grid, dotWidth, dotHeight, points[axis]!, points[(axis + 1) % 5]!)
    }
  }
  const profilePoints = values.map((value, axis) => pointOnAxis(center, {
    x: radii.x * clamp01(value),
    y: radii.y * clamp01(value),
  }, axis))
  for (let axis = 0; axis < 5; axis += 1) {
    drawLine(profile, dotWidth, dotHeight, profilePoints[axis]!, profilePoints[(axis + 1) % 5]!)
    setDot(profile, dotWidth, dotHeight, profilePoints[axis]!.x, profilePoints[axis]!.y)
  }

  const characters = Array.from({ length: safeRows }, () => Array(safeColumns).fill(" "))
  const tones = Array.from({ length: safeRows }, () =>
    Array<CatalogRadarTone>(safeColumns).fill("empty"))
  for (let y = 0; y < safeRows; y += 1) {
    for (let x = 0; x < safeColumns; x += 1) {
      const profileMask = brailleMask(profile, dotWidth, x, y)
      const gridMask = brailleMask(grid, dotWidth, x, y)
      const cell = catalogRadarCell(profileMask, gridMask)
      characters[y]![x] = cell.character
      tones[y]![x] = cell.tone
    }
  }

  const vertices = values.map((_, axis) => pointOnAxis(center, radii, axis))
  writeMetric(characters, tones, metrics[0], safeColumns / 2, 0, "center")
  for (const axis of [1, 2] as const) {
    const vertex = vertices[axis]
    writeMetric(
      characters,
      tones,
      metrics[axis],
      Math.floor(vertex.x / 2) + (axis === 1 ? upperLabelGap : lowerLabelGap),
      Math.max(2, Math.floor(vertex.y / 4) + (axis === 1 ? -2 : 0)),
      "left",
    )
  }
  for (const axis of [3, 4] as const) {
    const vertex = vertices[axis]
    writeMetric(
      characters,
      tones,
      metrics[axis],
      Math.ceil(vertex.x / 2) - (axis === 4 ? upperLabelGap : lowerLabelGap),
      Math.max(2, Math.floor(vertex.y / 4) + (axis === 4 ? -2 : 0)),
      "right",
    )
  }

  return characters.map((row, y) => {
    const runs: CatalogRadarRun[] = []
    for (let x = 0; x < safeColumns; x += 1) {
      const tone = tones[y]![x]!
      const character = row[x]!
      const previous = runs.at(-1)
      if (previous?.tone === tone) {
        runs[runs.length - 1] = { tone, text: `${previous.text}${character}` }
      } else {
        runs.push({ tone, text: character })
      }
    }
    return runs
  })
}

export const catalogRadarCell = (
  profileMask: number,
  gridMask: number,
): { readonly character: string; readonly tone: CatalogRadarTone } => {
  if (profileMask > 0) {
    return { character: String.fromCodePoint(0x2800 + (profileMask & 0xff)), tone: "profile" }
  }
  if (gridMask > 0) {
    return { character: String.fromCodePoint(0x2800 + (gridMask & 0xff)), tone: "grid" }
  }
  return { character: " ", tone: "empty" }
}
