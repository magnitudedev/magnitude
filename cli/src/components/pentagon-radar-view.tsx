import { memo, useState, useSyncExternalStore } from "react"
import { Option } from "effect"
import {
  getAnimationTimeSnapshot,
  subscribeAnimationClock,
  subscribeAnimationNoop,
} from "@magnitudedev/client-common"
import { useTheme } from "../hooks/use-theme"
import {
  PENTAGON_RADAR_DURATION_MS,
  pentagonRadarValues,
  pentagonRadarTransitionValues,
  renderPentagonRadar,
  retargetPentagonRadar,
  type PentagonRadarAxes,
  type PentagonRadarTransition,
  type PentagonRadarValues,
} from "./pentagon-radar"

interface PentagonRadarAnimation {
  readonly target: PentagonRadarValues
  readonly transition: PentagonRadarTransition | null
}

const radarValuesEqual = (
  left: PentagonRadarValues,
  right: PentagonRadarValues,
): boolean => left.every((value, index) => Option.match(value, {
  onNone: () => Option.isNone(right[index]!),
  onSome: (measurement) => Option.contains(right[index]!, measurement),
}))

const usePentagonRadarAnimation = (target: PentagonRadarValues): PentagonRadarAnimation => {
  const [animation, setAnimation] = useState<PentagonRadarAnimation>(() => ({
    target,
    transition: null,
  }))
  if (radarValuesEqual(animation.target, target)) return animation

  const next = {
    target,
    transition: retargetPentagonRadar(
      animation.target,
      target,
      animation.transition,
      getAnimationTimeSnapshot(),
    ),
  }
  setAnimation(next)
  return next
}

const useRadarAnimationTime = (transition: PentagonRadarTransition | null): number => {
  const endAt = transition === null
    ? 0
    : transition.startedAt + PENTAGON_RADAR_DURATION_MS
  const active = transition !== null && getAnimationTimeSnapshot() < endAt
  const endSnapshot = () => endAt
  return useSyncExternalStore(
    active ? subscribeAnimationClock : subscribeAnimationNoop,
    active ? getAnimationTimeSnapshot : endSnapshot,
    active ? getAnimationTimeSnapshot : endSnapshot,
  )
}

export const PentagonRadarView = memo(function PentagonRadarView({
  axes,
  columns,
}: {
  readonly axes: PentagonRadarAxes
  readonly columns?: number
}) {
  const theme = useTheme()
  const { transition } = usePentagonRadarAnimation(pentagonRadarValues(axes))
  const now = useRadarAnimationTime(transition)
  const values = transition !== null && now < transition.startedAt + PENTAGON_RADAR_DURATION_MS
    ? pentagonRadarTransitionValues(transition, now)
    : undefined
  const frame = renderPentagonRadar(axes, values, columns)
  return (
    <box style={{ flexDirection: "column", height: frame.length, minHeight: frame.length, flexShrink: 0 }}>
      {frame.map((row, rowIndex) => (
        <text key={rowIndex} wrapMode="none">
          {row.map((run, runIndex) => (
            <span
              key={runIndex}
              fg={run.tone === "profile"
                ? theme.accent
                : run.tone === "label"
                  ? theme.text.body
                  : run.tone === "detail"
                    ? theme.text.supporting
                    : theme.border.standard}
            >
              {run.text}
            </span>
          ))}
        </text>
      ))}
    </box>
  )
})
