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
    <div className="mt-2">
      <svg
        className="h-auto max-h-[340px] w-full overflow-visible max-[1050px]:max-h-[220px]"
        viewBox="0 0 360 320"
        role="img"
        aria-label={`Model comparison profile. ${description}`}
      >
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <polygon
            key={scale}
            className="fill-none stroke-slate-300 stroke-1 [vector-effect:non-scaling-stroke] dark:stroke-slate-750"
            points={chartPoints([1, 1, 1, 1, 1], scale)}
          />
        ))}
        {[0, 1, 2, 3, 4].map((index) => {
          const outer = pointOnAxis(index, 1, centerX, centerY, radius)
          return (
            <line
              key={index}
              className="fill-none stroke-slate-300 stroke-1 [vector-effect:non-scaling-stroke] dark:stroke-slate-750"
              x1={centerX}
              y1={centerY}
              x2={outer.x}
              y2={outer.y}
            />
          )
        })}
        <polygon
          className="fill-blue-700/20 stroke-blue-700 stroke-2 [stroke-linejoin:round] [vector-effect:non-scaling-stroke] dark:fill-blue-500/20 dark:stroke-blue-500"
          points={chartPoints(values)}
        />
        {values.map((value, index) => {
          const point = pointOnAxis(index, value, centerX, centerY, radius)
          return (
            <circle
              key={axes[index].label}
              className="fill-white stroke-violet-700 stroke-2 [vector-effect:non-scaling-stroke] dark:fill-slate-850 dark:stroke-violet-500"
              cx={point.x}
              cy={point.y}
              r="3.2"
            />
          )
        })}
        {axes.map(({ label, detail, value }, index) => {
          const unavailable = Option.isNone(value)
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
              className="fill-slate-600 font-sans text-[9px] font-[650] leading-[normal] tracking-[.055em] dark:fill-slate-400"
              x={point.x}
              y={point.y}
              textAnchor={anchor}
            >
              <tspan x={point.x}>{label}</tspan>
              <tspan
                className={`${
                  unavailable
                    ? "fill-slate-500"
                    : "fill-slate-900 dark:fill-slate-200"
                } font-mono text-[10px] font-normal leading-[normal] tracking-normal`}
                x={point.x}
                dy="14"
              >
                {detail}
              </tspan>
            </text>
          )
        })}
      </svg>
    </div>
  )
}
