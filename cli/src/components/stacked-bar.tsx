const PARTIAL_BLOCKS = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'] as const
const EIGHTHS_PER_CELL = 8

export interface StackedBarSegment {
  readonly value: number
  readonly color: string
  readonly fill?: 'solid' | 'shade'
}

export interface StackedBarCell {
  readonly character: string
  readonly foreground: string
  readonly background?: string
}

interface AllocatedRun extends StackedBarSegment {
  readonly start: number
  readonly end: number
}

const allocateEighths = (values: readonly number[], total: number, units: number): readonly number[] => {
  if (total <= 0 || units <= 0) return values.map(() => 0)
  let remainingValue = total
  const bounded = values.map((value) => {
    const accepted = Math.min(remainingValue, Math.max(0, Number.isFinite(value) ? value : 0))
    remainingValue -= accepted
    return accepted
  })
  const exact = bounded.map((value) => value / total * units)
  const allocated = exact.map(Math.floor)
  let remainingUnits = units - allocated.reduce((sum, value) => sum + value, 0)
  const order = Arr.zip(exact, allocated)
    .map(([value, allocatedValue], index) => ({ index, remainder: value - allocatedValue }))
    .sort((left, right) => right.remainder - left.remainder || left.index - right.index)
  for (const candidate of order) {
    if (remainingUnits <= 0) break
    const current = Option.getOrElse(Arr.get(allocated, candidate.index), () => 0)
    allocated.splice(candidate.index, 1, current + 1)
    remainingUnits -= 1
  }
  return allocated
}

/**
 * Projects a stacked value series into terminal cells. A boundary cell uses the partial-block
 * glyph as its foreground and the following segment as its background, giving transitions
 * one-eighth-of-a-cell precision without a visible gap.
 */
export const createStackedBarCells = (
  segments: readonly StackedBarSegment[],
  total: number,
  width: number,
  track: StackedBarSegment,
): readonly StackedBarCell[] => {
  if (width <= 0 || total <= 0) return []
  const segmentValue = segments.reduce(
    (sum, segment) => sum + Math.max(0, Number.isFinite(segment.value) ? segment.value : 0),
    0,
  )
  const visibleSegments = [
    ...segments,
    { ...track, value: Math.max(0, total - Math.min(total, segmentValue)) },
  ]
  const units = allocateEighths(
    visibleSegments.map((segment) => segment.value),
    total,
    width * EIGHTHS_PER_CELL,
  )
  const runs: AllocatedRun[] = []
  let cursor = 0
  for (const [segment, length] of Arr.zip(visibleSegments, units)) {
    if (length === 0) continue
    runs.push({ ...segment, start: cursor, end: cursor + length })
    cursor += length
  }
  if (runs.length === 0) return []

  const runAt = (unit: number): AllocatedRun => {
    return Option.getOrThrowWith(
      Option.orElse(
        Arr.findFirst(runs, (candidate) => unit >= candidate.start && unit < candidate.end),
        () => Arr.last(runs),
      ),
      () => new Error('stacked bar has no allocated run'),
    )
  }

  return Array.from({ length: width }, (_, index): StackedBarCell => {
    const start = index * EIGHTHS_PER_CELL
    const end = start + EIGHTHS_PER_CELL
    const left = runAt(start)
    const right = runAt(end - 1)
    if (left === right) {
      return {
        character: left.fill === 'shade' ? '░' : '█',
        foreground: left.color,
      }
    }
    const foregroundEighths = Math.max(1, Math.min(EIGHTHS_PER_CELL - 1, left.end - start))
    const character = Option.getOrThrowWith(
      Arr.get(PARTIAL_BLOCKS, foregroundEighths),
      () => new Error(`invalid partial-block width: ${foregroundEighths}`),
    )
    return {
      character,
      foreground: left.color,
      background: right.color,
    }
  })
}

interface StackedBarProps {
  readonly segments: readonly StackedBarSegment[]
  readonly total: number
  readonly width: number
  readonly trackColor: string
  readonly trackFill?: StackedBarSegment['fill']
}

export const StackedBar = ({
  segments,
  total,
  width,
  trackColor,
  trackFill = 'shade',
}: StackedBarProps) => {
  const cells = createStackedBarCells(
    segments,
    total,
    width,
    { value: 0, color: trackColor, fill: trackFill },
  )
  if (cells.length === 0) return null
  return (
    <text>
      {cells.map((cell, index) => (
        <span key={index} fg={cell.foreground} bg={cell.background}>{cell.character}</span>
      ))}
    </text>
  )
}
import { Array as Arr, Option } from 'effect'
