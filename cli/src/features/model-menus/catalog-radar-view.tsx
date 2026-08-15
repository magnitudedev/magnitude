import { memo } from "react"
import { Option } from "effect"
import {
  PENTAGON_RADAR_ROWS,
  type PentagonRadarAxes,
  type PentagonRadarTransition,
} from "../../components/pentagon-radar"
import { PentagonRadarView } from "../../components/pentagon-radar-view"
import { useTheme } from "../../hooks/use-theme"

export const CatalogRadarView = memo(function CatalogRadarView({
  axes,
  transition,
}: {
  readonly axes: Option.Option<PentagonRadarAxes>
  readonly transition: PentagonRadarTransition | null
}) {
  const theme = useTheme()
  return Option.match(axes, {
    onNone: () => (
      <box style={{ flexDirection: "column", height: PENTAGON_RADAR_ROWS, minHeight: PENTAGON_RADAR_ROWS, flexShrink: 0 }}>
        <text style={{ fg: theme.muted }}>Radar profile unavailable</text>
      </box>
    ),
    onSome: (value) => <PentagonRadarView axes={value} transition={transition} />,
  })
})
