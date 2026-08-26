import { Option } from "effect"

export interface RadarPoint {
  readonly x: number
  readonly y: number
}

export interface RadarGuides {
  readonly ringCount: number
  readonly spokes: boolean
}

export type RadarGuideConfiguration = false | RadarGuides
export type RadarTone = "empty" | "guide" | "profile"

export interface RadarCell {
  readonly character: string
  readonly tone: RadarTone
}

export interface RadarRun {
  readonly text: string
  readonly tone: RadarTone
}

export type RadarCellFrame = readonly (readonly RadarCell[])[]
export type RadarFrame = readonly (readonly RadarRun[])[]
export type RadarValues = readonly Option.Option<number>[]

export interface RadarRenderOptions {
  readonly columns: number
  readonly rows: number
  readonly guides: RadarGuideConfiguration
  readonly center?: RadarPoint
  readonly radius?: number
}

const BRAILLE_BITS = [
  [0x01, 0x08],
  [0x02, 0x10],
  [0x04, 0x20],
  [0x40, 0x80],
] as const

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value))

const positiveInteger = (value: number, name: string): number => {
  const integer = Math.floor(value)
  if (!Number.isFinite(value) || integer < 1) {
    throw new RangeError(`${name} must be a finite positive integer`)
  }
  return integer
}

const nonnegativeInteger = (value: number, name: string): number => {
  const integer = Math.floor(value)
  if (!Number.isFinite(value) || integer < 0) {
    throw new RangeError(`${name} must be a finite nonnegative integer`)
  }
  return integer
}

const finiteNonnegative = (value: number, name: string): number => {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be finite and nonnegative`)
  }
  return value
}

const pointOnAxis = (
  center: RadarPoint,
  radius: number,
  axis: number,
  axisCount: number,
): RadarPoint => {
  const angle = -Math.PI / 2 + axis * Math.PI * 2 / axisCount
  return {
    x: center.x + Math.cos(angle) * radius,
    y: center.y + Math.sin(angle) * radius,
  }
}

const setDot = (
  buffer: Uint8Array,
  dotWidth: number,
  dotHeight: number,
  x: number,
  y: number,
): void => {
  const roundedX = Math.round(x)
  const roundedY = Math.round(y)
  if (roundedX < 0 || roundedY < 0 || roundedX >= dotWidth || roundedY >= dotHeight) return
  buffer[roundedY * dotWidth + roundedX] = 1
}

const drawLine = (
  buffer: Uint8Array,
  dotWidth: number,
  dotHeight: number,
  start: RadarPoint,
  end: RadarPoint,
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

export const radarCell = (profileMask: number, guideMask: number): RadarCell => {
  if (profileMask > 0) {
    return { character: String.fromCodePoint(0x2800 + (profileMask & 0xff)), tone: "profile" }
  }
  if (guideMask > 0) {
    return { character: String.fromCodePoint(0x2800 + (guideMask & 0xff)), tone: "guide" }
  }
  return { character: " ", tone: "empty" }
}

export const renderRadarCells = (
  values: RadarValues,
  options: RadarRenderOptions,
): RadarCellFrame => {
  const columns = positiveInteger(options.columns, "columns")
  const rows = positiveInteger(options.rows, "rows")
  if (values.length < 3) throw new RangeError("radar profiles require at least three values")

  const dotWidth = columns * 2
  const dotHeight = rows * 4
  const center = options.center ?? { x: (dotWidth - 1) / 2, y: (dotHeight - 1) / 2 }
  if (!Number.isFinite(center.x) || !Number.isFinite(center.y)) {
    throw new RangeError("radar center must be finite")
  }
  const maximumRadius = Math.max(0, Math.min(
    center.x,
    center.y,
    dotWidth - 1 - center.x,
    dotHeight - 1 - center.y,
  ))
  const radius = options.radius === undefined
    ? maximumRadius
    : finiteNonnegative(options.radius, "radius")

  const guide = new Uint8Array(dotWidth * dotHeight)
  const profile = new Uint8Array(dotWidth * dotHeight)
  const axisCount = values.length

  if (options.guides !== false) {
    const ringCount = nonnegativeInteger(options.guides.ringCount, "ringCount")
    if (options.guides.spokes) {
      for (let axis = 0; axis < axisCount; axis += 1) {
        drawLine(guide, dotWidth, dotHeight, center, pointOnAxis(center, radius, axis, axisCount))
      }
    }
    for (let ring = 1; ring <= ringCount; ring += 1) {
      const ringRadius = radius * ring / ringCount
      const points = values.map((_, axis) => pointOnAxis(center, ringRadius, axis, axisCount))
      for (let axis = 0; axis < axisCount; axis += 1) {
        drawLine(guide, dotWidth, dotHeight, points[axis]!, points[(axis + 1) % axisCount]!)
      }
    }
  }

  const profilePoints = values.map((value, axis) => Option.map(value, (measurement) => {
    if (!Number.isFinite(measurement)) throw new RangeError("radar measurements must be finite")
    return pointOnAxis(center, radius * clamp01(measurement), axis, axisCount)
  }))
  for (let axis = 0; axis < axisCount; axis += 1) {
    const point = profilePoints[axis]!
    const next = profilePoints[(axis + 1) % axisCount]!
    if (Option.isSome(point) && Option.isSome(next)) {
      drawLine(profile, dotWidth, dotHeight, point.value, next.value)
    }
    if (Option.isSome(point)) setDot(profile, dotWidth, dotHeight, point.value.x, point.value.y)
  }

  return Array.from({ length: rows }, (_, y) =>
    Array.from({ length: columns }, (_, x) => radarCell(
      brailleMask(profile, dotWidth, x, y),
      brailleMask(guide, dotWidth, x, y),
    )))
}

export const radarRuns = (frame: RadarCellFrame): RadarFrame => frame.map((row) => {
  const runs: RadarRun[] = []
  for (const cell of row) {
    const previous = runs.at(-1)
    if (previous?.tone === cell.tone) {
      runs[runs.length - 1] = { tone: cell.tone, text: `${previous.text}${cell.character}` }
    } else {
      runs.push({ tone: cell.tone, text: cell.character })
    }
  }
  return runs
})

export const renderRadar = (
  values: RadarValues,
  options: RadarRenderOptions,
): RadarFrame => radarRuns(renderRadarCells(values, options))

export const interpolateRadarValues = (
  from: RadarValues,
  to: RadarValues,
  progress: number,
): RadarValues => {
  if (from.length !== to.length) throw new RangeError("radar transitions require equal profile lengths")
  const clampedProgress = clamp01(progress)
  return from.map((value, index) => {
    const next = to[index]!
    if (Option.isNone(value) || Option.isNone(next)) return next
    return Option.some(value.value + (next.value - value.value) * clampedProgress)
  })
}

const mixedUint32 = (value: number): number => {
  let mixed = value >>> 0
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x21f0aaad)
  mixed = Math.imul(mixed ^ (mixed >>> 15), 0x735a2d97)
  return (mixed ^ (mixed >>> 15)) >>> 0
}

export const generateRadarValues = ({
  seed,
  targetIndex,
  pointCount,
  valueRange,
}: {
  readonly seed: number
  readonly targetIndex: number
  readonly pointCount: number
  readonly valueRange: readonly [number, number]
}): RadarValues => {
  const count = positiveInteger(pointCount, "pointCount")
  if (count < 3) throw new RangeError("pointCount must be at least three")
  if (!Number.isFinite(seed) || !Number.isFinite(targetIndex) || targetIndex < 0) {
    throw new RangeError("radar generator seed and target index must be finite and nonnegative")
  }
  const [minimum, maximum] = valueRange
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)
    || minimum < 0 || maximum > 1 || minimum > maximum) {
    throw new RangeError("valueRange must be ordered within [0, 1]")
  }
  return Array.from({ length: count }, (_, pointIndex) => {
    const input = (seed >>> 0)
      ^ Math.imul((Math.floor(targetIndex) + 1) >>> 0, 0x9e3779b1)
      ^ Math.imul(pointIndex + 1, 0x85ebca6b)
    const sample = mixedUint32(input) / 0xffff_ffff
    return Option.some(minimum + (maximum - minimum) * sample)
  })
}
