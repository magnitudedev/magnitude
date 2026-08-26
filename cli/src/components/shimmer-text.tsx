import { useTheme } from '../hooks/use-theme'
import { interpolateHexColor } from '@magnitudedev/client-common'
import { useAnimationTime } from '../hooks/use-animation-time'

const DEFAULT_SWEEP_DURATION_MS = 850
const DEFAULT_SWEEP_CYCLE_MS = 1_800
const SWEEP_RADIUS_RATIO = 0.22
const MIN_SWEEP_RADIUS = 2

export const sweepShimmerIntensities = ({
  length,
  timeMs,
  sweepDurationMs = DEFAULT_SWEEP_DURATION_MS,
  cycleDurationMs = DEFAULT_SWEEP_CYCLE_MS,
  radius,
}: {
  readonly length: number
  readonly timeMs: number
  readonly sweepDurationMs?: number
  readonly cycleDurationMs?: number
  readonly radius?: number
}): readonly number[] => {
  const safeLength = Math.max(0, Math.floor(length))
  const resolvedRadius = radius ?? Math.max(MIN_SWEEP_RADIUS, safeLength * SWEEP_RADIUS_RATIO)
  if (safeLength === 0 || sweepDurationMs <= 0 || cycleDurationMs <= 0 || resolvedRadius <= 0) return []
  const cycleTime = ((timeMs % cycleDurationMs) + cycleDurationMs) % cycleDurationMs
  if (cycleTime >= sweepDurationMs) return Array.from({ length: safeLength }, () => 0)
  const progress = cycleTime / sweepDurationMs
  const easedProgress = 0.5 - 0.5 * Math.cos(progress * Math.PI)
  const center = -resolvedRadius + easedProgress * (safeLength - 1 + resolvedRadius * 2)
  return Array.from({ length: safeLength }, (_, index) => {
    const distance = Math.abs(index - center)
    return distance >= resolvedRadius
      ? 0
      : 0.5 + 0.5 * Math.cos(Math.PI * distance / resolvedRadius)
  })
}

export const ShimmerText = ({
  text,
  baseColor,
  highlightColor,
  sweepDurationMs = DEFAULT_SWEEP_DURATION_MS,
  cycleDurationMs = DEFAULT_SWEEP_CYCLE_MS,
}: {
  readonly text: string
  readonly baseColor?: string
  readonly highlightColor?: string
  readonly sweepDurationMs?: number
  readonly cycleDurationMs?: number
}) => {
  const theme = useTheme()
  const animationTime = useAnimationTime(true)
  const resolvedBaseColor = baseColor ?? theme.text.supporting
  const resolvedHighlightColor = highlightColor ?? theme.text.emphasized
  const intensities = sweepShimmerIntensities({
    length: text.length,
    timeMs: animationTime,
    sweepDurationMs,
    cycleDurationMs,
  })
  if (intensities.every((intensity) => intensity === 0)) {
    return <span fg={resolvedBaseColor}>{text}</span>
  }
  return (
    <>
      {[...text].map((character, index) => (
        <span
          key={index}
          fg={interpolateHexColor(
            resolvedBaseColor,
            resolvedHighlightColor,
            intensities[index] ?? 0,
          )}
        >
          {character}
        </span>
      ))}
    </>
  )
}
