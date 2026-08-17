import { Option } from "effect"
import type { ReactNode } from "react"
import type { LocalModelRadarAxes } from "@magnitudedev/client-common"

interface Point {
  readonly x: number
  readonly y: number
}

const pointOnAxis = (
  index: number,
  value: number,
  centerX = 120,
  centerY = centerX,
  radius = 92
): Point => {
  const angle = -Math.PI / 2 + (index * Math.PI * 2) / 5
  return {
    x: centerX + Math.cos(angle) * radius * value,
    y: centerY + Math.sin(angle) * radius * value,
  }
}

export const radarPolygonPoints = (
  values: readonly number[],
  scale = 1
): string =>
  values
    .map((value, index) => pointOnAxis(index, value * scale))
    .map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" ")

export function ModelRadarChart({
  axes,
}: {
  readonly axes: LocalModelRadarAxes
}): ReactNode {
  const values = axes.map(({ value }) => Option.getOrElse(value, () => 0))
  const description = axes
    .map(({ label, detail }) => `${label.toLowerCase()}: ${detail}`)
    .join(", ")
  const centerX = 180
  const centerY = 158
  const radius = 88
  const labelRadius = 126
  const chartPoints = (chartValues: readonly number[], scale = 1) =>
    chartValues
      .map((value, index) =>
        pointOnAxis(index, value * scale, centerX, centerY, radius)
      )
      .map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`)
      .join(" ")

  return (
    <div className="mc-radar">
      <svg
        className="mc-radar-plot"
        viewBox="0 0 360 320"
        role="img"
        aria-label={`Model comparison profile. ${description}`}
      >
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <polygon
            key={scale}
            className="mc-radar-grid"
            points={chartPoints([1, 1, 1, 1, 1], scale)}
          />
        ))}
        {[0, 1, 2, 3, 4].map((index) => {
          const outer = pointOnAxis(index, 1, centerX, centerY, radius)
          return (
            <line
              key={index}
              className="mc-radar-grid"
              x1={centerX}
              y1={centerY}
              x2={outer.x}
              y2={outer.y}
            />
          )
        })}
        <polygon
          className="mc-radar-profile"
          points={chartPoints(values)}
        />
        {values.map((value, index) => {
          const point = pointOnAxis(index, value, centerX, centerY, radius)
          return (
            <circle
              key={axes[index].label}
              className="mc-radar-point"
              cx={point.x}
              cy={point.y}
              r="3.2"
            />
          )
        })}
        {axes.map(({ label, detail, value }, index) => {
          const point = pointOnAxis(index, 1, centerX, centerY, labelRadius)
          const anchor =
            point.x < centerX - 10
              ? "end"
              : point.x > centerX + 10
                ? "start"
                : "middle"
          return (
            <text
              key={label}
              className="mc-radar-axis-label"
              data-unavailable={Option.isNone(value)}
              x={point.x}
              y={point.y}
              textAnchor={anchor}
            >
              <tspan x={point.x}>{label}</tspan>
              <tspan className="mc-radar-axis-detail" x={point.x} dy="14">
                {detail}
              </tspan>
            </text>
          )
        })}
      </svg>
    </div>
  )
}
