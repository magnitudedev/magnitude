export const PENTAGON_RADAR_DURATION_MS = 220
export const PENTAGON_RADAR_COLUMNS = 56
export const PENTAGON_RADAR_ROWS = 15

export interface PentagonRadarAxis {
  readonly value: Option.Option<number>
  readonly label: string
  readonly detail: string
}

// Axis order follows the pentagon clockwise from its top vertex.
export type PentagonRadarAxes = readonly [
  PentagonRadarAxis,
  PentagonRadarAxis,
  PentagonRadarAxis,
  PentagonRadarAxis,
  PentagonRadarAxis,
]

export type PentagonRadarValues = readonly [
  Option.Option<number>,
  Option.Option<number>,
  Option.Option<number>,
  Option.Option<number>,
  Option.Option<number>,
]

export interface PentagonRadarTransition {
  readonly from: PentagonRadarValues
  readonly to: PentagonRadarValues
  readonly startedAt: number
}

export type PentagonRadarTone = "empty" | "grid" | "profile" | "label" | "detail"

export interface PentagonRadarRun {
  readonly text: string
  readonly tone: PentagonRadarTone
}

export type PentagonRadarFrame = readonly (readonly PentagonRadarRun[])[]

interface Point {
  readonly x: number
  readonly y: number
}

interface Radii {
  readonly x: number
  readonly y: number
}

type AxisAlignment = "left" | "center" | "right"

const BRAILLE_BITS = [
  [0x01, 0x08],
  [0x02, 0x10],
  [0x04, 0x20],
  [0x40, 0x80],
] as const

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value))

export const pentagonRadarValues = (axes: PentagonRadarAxes): PentagonRadarValues =>
  axes.map(({ value }) => Option.map(value, clamp01)) as unknown as PentagonRadarValues

export const pentagonRadarEase = (progress: number): number =>
  1 - (1 - clamp01(progress)) ** 3

export const interpolatePentagonRadar = (
  from: PentagonRadarValues,
  to: PentagonRadarValues,
  progress: number,
): PentagonRadarValues => {
  const eased = pentagonRadarEase(progress)
  return from.map((value, index) => {
    const next = to[index]!
    if (Option.isNone(next) || Option.isNone(value)) return next
    return Option.some(value.value + (next.value - value.value) * eased)
  }) as unknown as PentagonRadarValues
}

export const pentagonRadarTransitionValues = (
  transition: PentagonRadarTransition,
  now: number,
): PentagonRadarValues => interpolatePentagonRadar(
  transition.from,
  transition.to,
  (now - transition.startedAt) / PENTAGON_RADAR_DURATION_MS,
)

export const retargetPentagonRadar = (
  current: PentagonRadarValues,
  next: PentagonRadarValues,
  previous: PentagonRadarTransition | null,
  now: number,
): PentagonRadarTransition => ({
  from: previous === null ? current : pentagonRadarTransitionValues(previous, now),
  to: next,
  startedAt: now,
})

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

const writeText = (
  characters: string[][],
  tones: PentagonRadarTone[][],
  text: string,
  startX: number,
  y: number,
  tone: Extract<PentagonRadarTone, "label" | "detail">,
): void => {
  if (y < 0 || y >= characters.length) return
  for (let index = 0; index < text.length; index += 1) {
    const x = startX + index
    if (x < 0 || x >= characters[y]!.length) continue
    characters[y]![x] = text[index]!
    tones[y]![x] = tone
  }
}

const alignedStartX = (
  text: string,
  anchorX: number,
  alignment: AxisAlignment,
): number => alignment === "left"
  ? anchorX
  : alignment === "right"
    ? anchorX - text.length + 1
    : Math.round(anchorX - text.length / 2)

const writeAxis = (
  characters: string[][],
  tones: PentagonRadarTone[][],
  axis: PentagonRadarAxis,
  anchorX: number,
  y: number,
  alignment: AxisAlignment,
): void => {
  writeText(characters, tones, axis.label, alignedStartX(axis.label, anchorX, alignment), y, "label")
  writeText(characters, tones, axis.detail, alignedStartX(axis.detail, anchorX, alignment), y + 1, "detail")
}

export const pentagonRadarCell = (
  profileMask: number,
  gridMask: number,
): { readonly character: string; readonly tone: PentagonRadarTone } => {
  if (profileMask > 0) {
    return { character: String.fromCodePoint(0x2800 + (profileMask & 0xff)), tone: "profile" }
  }
  if (gridMask > 0) {
    return { character: String.fromCodePoint(0x2800 + (gridMask & 0xff)), tone: "grid" }
  }
  return { character: " ", tone: "empty" }
}

export const renderPentagonRadar = (
  axes: PentagonRadarAxes,
  values: PentagonRadarValues = pentagonRadarValues(axes),
  columns = PENTAGON_RADAR_COLUMNS,
  rows = PENTAGON_RADAR_ROWS,
): PentagonRadarFrame => {
  const safeColumns = Math.max(44, Math.floor(columns))
  const safeRows = Math.max(13, Math.floor(rows))
  const dotWidth = safeColumns * 2
  const dotHeight = safeRows * 4
  const chartTop = 8
  const radius = 24
  const upperLabelGap = 2
  const lowerLabelGap = 3
  const radii = { x: radius, y: radius }
  const center = { x: dotWidth / 2, y: chartTop + radius }
  const grid = new Uint8Array(dotWidth * dotHeight)
  const profile = new Uint8Array(dotWidth * dotHeight)

  for (let axis = 0; axis < axes.length; axis += 1) {
    drawLine(grid, dotWidth, dotHeight, center, pointOnAxis(center, radii, axis))
  }
  for (const scale of [1 / 3, 2 / 3, 1]) {
    const scaledRadii = { x: radii.x * scale, y: radii.y * scale }
    const points = axes.map((_, axis) => pointOnAxis(center, scaledRadii, axis))
    for (let axis = 0; axis < axes.length; axis += 1) {
      drawLine(grid, dotWidth, dotHeight, points[axis]!, points[(axis + 1) % axes.length]!)
    }
  }
  const profilePoints = values.map((value, axis) => Option.map(value, (measurement) =>
    pointOnAxis(center, {
      x: radii.x * clamp01(measurement),
      y: radii.y * clamp01(measurement),
    }, axis)))
  for (let axis = 0; axis < axes.length; axis += 1) {
    const point = profilePoints[axis]
    const next = profilePoints[(axis + 1) % axes.length]
    if (Option.isSome(point) && Option.isSome(next)) {
      drawLine(profile, dotWidth, dotHeight, point.value, next.value)
    }
    if (Option.isSome(point)) setDot(profile, dotWidth, dotHeight, point.value.x, point.value.y)
  }

  const characters = Array.from({ length: safeRows }, () => Array<string>(safeColumns).fill(" "))
  const tones = Array.from({ length: safeRows }, () =>
    Array<PentagonRadarTone>(safeColumns).fill("empty"))
  for (let y = 0; y < safeRows; y += 1) {
    for (let x = 0; x < safeColumns; x += 1) {
      const cell = pentagonRadarCell(
        brailleMask(profile, dotWidth, x, y),
        brailleMask(grid, dotWidth, x, y),
      )
      characters[y]![x] = cell.character
      tones[y]![x] = cell.tone
    }
  }

  const vertices = axes.map((_, axis) => pointOnAxis(center, radii, axis))
  writeAxis(characters, tones, axes[0], safeColumns / 2, 0, "center")
  for (const axis of [1, 2] as const) {
    const vertex = vertices[axis]
    writeAxis(
      characters,
      tones,
      axes[axis],
      Math.floor(vertex.x / 2) + (axis === 1 ? upperLabelGap : lowerLabelGap),
      Math.max(2, Math.floor(vertex.y / 4) + (axis === 1 ? -2 : 0)),
      "left",
    )
  }
  for (const axis of [3, 4] as const) {
    const vertex = vertices[axis]
    writeAxis(
      characters,
      tones,
      axes[axis],
      Math.ceil(vertex.x / 2) - (axis === 4 ? upperLabelGap : lowerLabelGap),
      Math.max(2, Math.floor(vertex.y / 4) + (axis === 4 ? -2 : 0)),
      "right",
    )
  }

  return characters.map((row, y) => {
    const runs: PentagonRadarRun[] = []
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
import { Option } from "effect"
