import { memo, useSyncExternalStore } from "react"
import {
  getAnimationTimeSnapshot,
  subscribeAnimationClock,
  subscribeAnimationNoop,
} from "@magnitudedev/client-common"
import { Option } from "effect"
import { useTheme } from "../../hooks/use-theme"
import {
  CATALOG_RADAR_DURATION_MS,
  CATALOG_RADAR_ROWS,
  catalogRadarTransitionValues,
  renderCatalogRadar,
  type CatalogRadarProfile,
  type CatalogRadarTransition,
} from "./catalog-radar"

const useRadarAnimationTime = (transition: CatalogRadarTransition | null): number => {
  const endAt = transition === null
    ? 0
    : transition.startedAt + CATALOG_RADAR_DURATION_MS
  const active = transition !== null && getAnimationTimeSnapshot() < endAt
  const endSnapshot = () => endAt
  return useSyncExternalStore(
    active ? subscribeAnimationClock : subscribeAnimationNoop,
    active ? getAnimationTimeSnapshot : endSnapshot,
    active ? getAnimationTimeSnapshot : endSnapshot,
  )
}

export const CatalogRadarView = memo(function CatalogRadarView({
  profile,
  transition,
}: {
  readonly profile: Option.Option<CatalogRadarProfile>
  readonly transition: CatalogRadarTransition | null
}) {
  const theme = useTheme()
  const now = useRadarAnimationTime(transition)
  if (Option.isNone(profile)) {
    return (
      <box style={{ flexDirection: "column", height: CATALOG_RADAR_ROWS, minHeight: CATALOG_RADAR_ROWS, flexShrink: 0 }}>
        <text style={{ fg: theme.muted }}>Radar profile unavailable</text>
      </box>
    )
  }
  const values = transition !== null && now < transition.startedAt + CATALOG_RADAR_DURATION_MS
    ? catalogRadarTransitionValues(transition, now)
    : profile.value.values
  const frame = renderCatalogRadar(values)
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
