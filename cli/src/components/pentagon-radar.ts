import { Option } from "effect"
import {
  interpolateRadarValues,
  renderRadarCells,
  type RadarTone,
} from "./radar"

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

type AxisAlignment = "left" | "center" | "right"

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value))

export const pentagonRadarValues = (axes: PentagonRadarAxes): PentagonRadarValues => [
  Option.map(axes[0].value, clamp01),
  Option.map(axes[1].value, clamp01),
  Option.map(axes[2].value, clamp01),
  Option.map(axes[3].value, clamp01),
  Option.map(axes[4].value, clamp01),
]

export const pentagonRadarEase = (progress: number): number =>
  1 - (1 - clamp01(progress)) ** 3

const asPentagonValues = (
  values: ReturnType<typeof interpolateRadarValues>,
): PentagonRadarValues => [
  values[0]!,
  values[1]!,
  values[2]!,
  values[3]!,
  values[4]!,
]

export const interpolatePentagonRadar = (
  from: PentagonRadarValues,
  to: PentagonRadarValues,
  progress: number,
): PentagonRadarValues => asPentagonValues(interpolateRadarValues(
  from,
  to,
  pentagonRadarEase(progress),
))

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

const pointOnAxis = (center: Point, radius: number, axis: number): Point => {
  const angle = -Math.PI / 2 + axis * Math.PI * 2 / 5
  return {
    x: center.x + Math.cos(angle) * radius,
    y: center.y + Math.sin(angle) * radius,
  }
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

const pentagonTone = (tone: RadarTone): PentagonRadarTone => tone === "guide" ? "grid" : tone

export const renderPentagonRadar = (
  axes: PentagonRadarAxes,
  values: PentagonRadarValues = pentagonRadarValues(axes),
  columns = PENTAGON_RADAR_COLUMNS,
  rows = PENTAGON_RADAR_ROWS,
): PentagonRadarFrame => {
  const safeColumns = Math.max(44, Math.floor(columns))
  const safeRows = Math.max(13, Math.floor(rows))
  const dotWidth = safeColumns * 2
  const chartTop = 8
  const radius = 24
  const upperLabelGap = 2
  const lowerLabelGap = 3
  const center = { x: dotWidth / 2, y: chartTop + radius }
  const cells = renderRadarCells(values, {
    columns: safeColumns,
    rows: safeRows,
    guides: { ringCount: 3, spokes: true },
    center,
    radius,
  })
  const characters = cells.map((row) => row.map(({ character }) => character))
  const tones = cells.map((row) => row.map(({ tone }) => pentagonTone(tone)))

  const vertices = axes.map((_, axis) => pointOnAxis(center, radius, axis))
  writeAxis(characters, tones, axes[0], safeColumns / 2, 0, "center")
  for (const axis of [1, 2] as const) {
    const vertex = vertices[axis]!
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
    const vertex = vertices[axis]!
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
