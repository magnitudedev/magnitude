import { memo } from "react"
import { Option } from "effect"
import {
  PENTAGON_RADAR_ROWS,
  type PentagonRadarAxes,
  type PentagonRadarTransition,
} from "../../components/pentagon-radar"
import { PentagonRadarView } from "../../components/pentagon-radar-view"

export const CatalogRadarView = memo(function CatalogRadarView({
  axes,
  transition,
}: {
  readonly axes: Option.Option<PentagonRadarAxes>
  readonly transition: PentagonRadarTransition | null
}) {
  return Option.match(axes, {
    onNone: () => (
      <box style={{ height: PENTAGON_RADAR_ROWS, minHeight: PENTAGON_RADAR_ROWS, flexShrink: 0 }} />
    ),
    onSome: (value) => <PentagonRadarView axes={value} transition={transition} />,
  })
})
