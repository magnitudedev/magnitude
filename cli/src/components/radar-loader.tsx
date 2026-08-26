import { useState } from "react"
import { interpolateHexColor } from "@magnitudedev/client-common"
import {
  generateRadarValues,
  interpolateRadarValues,
  renderRadarCells,
  type RadarGuideConfiguration,
  type RadarValues,
} from "./radar"
import { useAnimationTime } from "../hooks/use-animation-time"
import { useTheme } from "../hooks/use-theme"

const DEFAULT_COLUMNS = 20
const DEFAULT_ROWS = 10
const DEFAULT_POINT_COUNT = 5
const DEFAULT_INTERVAL_MS = 800
const DEFAULT_VALUE_RANGE = [0.25, 1] as const
const DEFAULT_GUIDES = false

export interface RadarLoaderProps {
  readonly columns?: number
  readonly rows?: number
  readonly pointCount?: number
  readonly guides?: RadarGuideConfiguration
  readonly intervalMs?: number
  readonly transitionMs?: number
  readonly valueRange?: readonly [number, number]
}

const positiveDuration = (value: number, name: string): number => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be finite and greater than zero`)
  }
  return value
}

export const radarLoaderEase = (progress: number): number => {
  const clamped = Math.min(1, Math.max(0, progress))
  return 0.5 - 0.5 * Math.cos(clamped * Math.PI)
}

export const radarLoaderPulseIntensity = ({
  timeMs,
  epochMs,
  movementPeriodMs,
  position,
}: {
  readonly timeMs: number
  readonly epochMs: number
  readonly movementPeriodMs: number
  readonly position: number
}): number => {
  const movementPeriod = positiveDuration(movementPeriodMs, "movementPeriodMs")
  if (!Number.isFinite(timeMs) || !Number.isFinite(epochMs) || !Number.isFinite(position)) {
    throw new RangeError("radar loader time and pulse position must be finite")
  }
  const pulsePeriod = movementPeriod * 2
  const elapsed = Math.max(0, timeMs - epochMs)
  const pulsePosition = (elapsed % pulsePeriod) / pulsePeriod
  const normalizedPosition = ((position % 1) + 1) % 1
  const directDistance = Math.abs(normalizedPosition - pulsePosition)
  const wrappedDistance = Math.min(directDistance, 1 - directDistance)
  return 0.5 + 0.5 * Math.cos(wrappedDistance * Math.PI * 2)
}

const radarCellPosition = (
  column: number,
  row: number,
  columns: number,
  rows: number,
): number => {
  const centerX = (columns * 2 - 1) / 2
  const centerY = (rows * 4 - 1) / 2
  const cellX = column * 2 + 0.5
  const cellY = row * 4 + 1.5
  const angleFromTop = Math.atan2(cellY - centerY, cellX - centerX) + Math.PI / 2
  return ((angleFromTop / (Math.PI * 2)) % 1 + 1) % 1
}

export const radarLoaderValues = ({
  timeMs,
  epochMs,
  seed,
  pointCount,
  intervalMs,
  transitionMs,
  valueRange,
}: {
  readonly timeMs: number
  readonly epochMs: number
  readonly seed: number
  readonly pointCount: number
  readonly intervalMs: number
  readonly transitionMs: number
  readonly valueRange: readonly [number, number]
}): RadarValues => {
  const interval = positiveDuration(intervalMs, "intervalMs")
  const transition = positiveDuration(transitionMs, "transitionMs")
  if (transition > interval) throw new RangeError("transitionMs must not exceed intervalMs")
  if (!Number.isFinite(timeMs) || !Number.isFinite(epochMs)) {
    throw new RangeError("radar loader time must be finite")
  }

  const elapsed = Math.max(0, timeMs - epochMs)
  const targetIndex = Math.floor(elapsed / interval)
  const withinInterval = elapsed - targetIndex * interval
  const progress = radarLoaderEase(withinInterval / transition)
  const from = generateRadarValues({ seed, targetIndex, pointCount, valueRange })
  const to = generateRadarValues({ seed, targetIndex: targetIndex + 1, pointCount, valueRange })
  return interpolateRadarValues(from, to, progress)
}

export const RadarLoader = ({
  columns = DEFAULT_COLUMNS,
  rows = DEFAULT_ROWS,
  pointCount = DEFAULT_POINT_COUNT,
  guides = DEFAULT_GUIDES,
  intervalMs = DEFAULT_INTERVAL_MS,
  transitionMs = intervalMs,
  valueRange = DEFAULT_VALUE_RANGE,
}: RadarLoaderProps) => {
  const theme = useTheme()
  const now = useAnimationTime(true)
  const [epochMs] = useState(now)
  const [seed] = useState(() => Math.floor(Math.random() * 0x1_0000_0000))
  const values = radarLoaderValues({
    timeMs: now,
    epochMs,
    seed,
    pointCount,
    intervalMs,
    transitionMs,
    valueRange,
  })
  const frame = renderRadarCells(values, { columns, rows, guides })
  const frameRows = frame.length
  const frameColumns = frame[0]?.length ?? 0

  return (
    <box style={{ flexDirection: "column", width: frameColumns, height: frameRows, flexShrink: 0 }}>
      {frame.map((row, rowIndex) => (
        <text key={rowIndex} wrapMode="none">
          {row.map((cell, columnIndex) => (
            <span
              key={columnIndex}
              fg={cell.tone === "profile"
                ? interpolateHexColor(
                  theme.text.emphasized,
                  theme.accent,
                  radarLoaderPulseIntensity({
                    timeMs: now,
                    epochMs,
                    movementPeriodMs: intervalMs,
                    position: radarCellPosition(columnIndex, rowIndex, frameColumns, frameRows),
                  }),
                )
                : theme.border.standard}
            >
              {cell.character}
            </span>
          ))}
        </text>
      ))}
    </box>
  )
}
