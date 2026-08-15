import { Option } from "effect"
import type { LocalModel } from "@magnitudedev/sdk"

export const CATALOG_RADAR_DURATION_MS = 220
export const CATALOG_RADAR_COLUMNS = 26
export const CATALOG_RADAR_ROWS = 9

// Axis order follows the triangle clockwise: intelligence, speed, accuracy.
export type CatalogRadarValues = readonly [number, number, number]

export interface CatalogRadarProfile {
  readonly values: CatalogRadarValues
}

export interface CatalogRadarTransition {
  readonly from: CatalogRadarValues
  readonly to: CatalogRadarValues
  readonly startedAt: number
}

export type CatalogRadarTone = "empty" | "grid" | "profile" | "label"

export interface CatalogRadarRun {
  readonly text: string
  readonly tone: CatalogRadarTone
}

export type CatalogRadarFrame = readonly (readonly CatalogRadarRun[])[]

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value))

export const normalizeCatalogRadarSpeed = (tokensPerSecond: number): number =>
  clamp01(Math.log(tokensPerSecond / 5) / Math.log(60 / 5))

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
    clamp01(catalog.fidelityRank / 100),
  ]
  if (values.some((value) => !Number.isFinite(value))) return Option.none()

  return Option.some({
    values,
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

const BRAILLE_BITS = [
  [0x01, 0x08],
  [0x02, 0x10],
  [0x04, 0x20],
  [0x40, 0x80],
] as const

const pointOnAxis = (center: Point, radius: number, axis: number): Point => {
  const angle = -Math.PI / 2 + axis * Math.PI * 2 / 3
  return {
    x: center.x + Math.cos(angle) * radius,
    y: center.y + Math.sin(angle) * radius,
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
): void => {
  if (y < 0 || y >= characters.length) return
  for (let index = 0; index < text.length; index += 1) {
    const x = startX + index
    if (x < 0 || x >= characters[y]!.length) continue
    characters[y]![x] = text[index]!
    tones[y]![x] = "label"
  }
}

export const renderCatalogRadar = (
  values: CatalogRadarValues,
  columns = CATALOG_RADAR_COLUMNS,
  rows = CATALOG_RADAR_ROWS,
): CatalogRadarFrame => {
  const safeColumns = Math.max(22, Math.floor(columns))
  const safeRows = Math.max(7, Math.floor(rows))
  const dotWidth = safeColumns * 2
  const dotHeight = safeRows * 4
  const chartTop = 4
  const chartBottom = (safeRows - 1) * 4 - 1
  const verticalRadius = (chartBottom - chartTop) * 2 / 3
  const radius = Math.max(7, Math.min(verticalRadius, dotWidth / 2 - 4))
  // Anchor the base to the bottom Braille dots of the final chart row.
  const center = {
    x: dotWidth / 2,
    y: chartBottom - radius / 2,
  }
  const grid = new Uint8Array(dotWidth * dotHeight)
  const profile = new Uint8Array(dotWidth * dotHeight)

  for (let axis = 0; axis < 3; axis += 1) {
    drawLine(grid, dotWidth, dotHeight, center, pointOnAxis(center, radius, axis))
  }
  for (const scale of [1 / 2, 1]) {
    const points = values.map((_, axis) => pointOnAxis(center, radius * scale, axis))
    for (let axis = 0; axis < 3; axis += 1) {
      drawLine(grid, dotWidth, dotHeight, points[axis]!, points[(axis + 1) % 3]!)
    }
  }
  const profilePoints = values.map((value, axis) =>
    pointOnAxis(center, radius * clamp01(value), axis))
  for (let axis = 0; axis < 3; axis += 1) {
    drawLine(profile, dotWidth, dotHeight, profilePoints[axis]!, profilePoints[(axis + 1) % 3]!)
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

  const labels = ["INT", "SPD", "ACC"] as const
  const centerColumn = safeColumns / 2
  const lowerVertexOffset = Math.cos(Math.PI / 6) * radius / 2
  writeLabel(characters, tones, labels[0], Math.round(safeColumns / 2 - labels[0].length / 2), 0)
  writeLabel(characters, tones, labels[1], Math.round(centerColumn + lowerVertexOffset - labels[1].length / 2) + 2, safeRows - 1)
  writeLabel(characters, tones, labels[2], Math.round(centerColumn - lowerVertexOffset - labels[2].length / 2) - 2, safeRows - 1)

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
