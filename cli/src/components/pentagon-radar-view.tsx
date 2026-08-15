import { memo, useSyncExternalStore } from "react"
import {
  getAnimationTimeSnapshot,
  subscribeAnimationClock,
  subscribeAnimationNoop,
} from "@magnitudedev/client-common"
import { useTheme } from "../hooks/use-theme"
import {
  PENTAGON_RADAR_DURATION_MS,
  pentagonRadarTransitionValues,
  renderPentagonRadar,
  type PentagonRadarAxes,
  type PentagonRadarTransition,
} from "./pentagon-radar"

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
  transition,
}: {
  readonly axes: PentagonRadarAxes
  readonly transition: PentagonRadarTransition | null
}) {
  const theme = useTheme()
  const now = useRadarAnimationTime(transition)
  const values = transition !== null && now < transition.startedAt + PENTAGON_RADAR_DURATION_MS
    ? pentagonRadarTransitionValues(transition, now)
    : undefined
  const frame = renderPentagonRadar(axes, values)
  return (
    <box style={{ flexDirection: "column", height: frame.length, minHeight: frame.length, flexShrink: 0 }}>
      {frame.map((row, rowIndex) => (
        <text key={rowIndex} wrapMode="none">
          {row.map((run, runIndex) => (
            <span
              key={runIndex}
              fg={run.tone === "profile"
                ? theme.primary
                : run.tone === "label"
                  ? theme.foreground
                  : run.tone === "detail"
                    ? theme.muted
                    : theme.border}
            >
              {run.text}
            </span>
          ))}
        </text>
      ))}
    </box>
  )
})
