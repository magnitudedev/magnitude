import { useState, type ReactNode } from "react"
import { formatTokensCompact } from "@magnitudedev/client-common"
import type { ContextUsageDisplay } from "@magnitudedev/sdk"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

export interface ContextUsageIndicatorProps {
  context: ContextUsageDisplay | null
  tokenCap?: number | null
  size?: number
  strokeWidth?: number
  showTokenLabel?: boolean
  tooltip?: "popover" | "native" | "none"
  tooltipPlacement?: "above-right" | "above-center"
}
export function contextUsagePercent(
  context: ContextUsageDisplay | null,
  tokenCap: number | null | undefined
): number | null {
  const tokenEstimate = context?.tokenEstimate ?? null
  if (tokenEstimate === null || !tokenCap || tokenCap <= 0) return null
  return Math.min(100, Math.max(0, (tokenEstimate / tokenCap) * 100))
}
export function contextUsageTooltipLines(
  context: ContextUsageDisplay | null,
  tokenCap: number | null | undefined
): readonly [string, string, string] {
  const tokenEstimate = context?.tokenEstimate ?? null
  const heading = context?.isCompacting ? "Compacting..." : "Context"
  const hasCap = tokenCap !== null && tokenCap !== undefined && tokenCap > 0
  const tokens = `${tokenEstimate === null ? "-" : formatTokensCompact(tokenEstimate)} / ${
    hasCap ? formatTokensCompact(tokenCap) : "-"
  } tokens`
  const remaining =
    tokenEstimate === null
      ? "100% remaining"
      : hasCap
      ? `${Math.max(
          0,
          Math.min(100, 100 - Math.round((tokenEstimate / tokenCap) * 100))
        )}% remaining`
      : "--% remaining"
  return [heading, tokens, remaining]
}

export function contextUsageStrokeClass(percent: number | null, isCompacting: boolean): string {
  if (isCompacting) return "stroke-violet-700 dark:stroke-violet-400"
  if (percent !== null && percent >= 90) return "stroke-red-600 dark:stroke-red-400"
  if (percent !== null && percent >= 70) return "stroke-orange-700 dark:stroke-orange-400"
  return "stroke-blue-700 dark:stroke-blue-500"
}
export function ContextUsageIndicator({
  context,
  tokenCap,
  size = 18,
  strokeWidth = 1.8,
  showTokenLabel = false,
  tooltip = "popover",
  tooltipPlacement = "above-right",
}: ContextUsageIndicatorProps): ReactNode {
  const [active, setActive] = useState(false)
  const isCompacting = context?.isCompacting ?? false
  const tokenEstimate = context?.tokenEstimate ?? null
  const pct = contextUsagePercent(context, tokenCap)
  const tooltipLines = contextUsageTooltipLines(context, tokenCap)
  const accessibleLabel = tooltipLines.join(" ")
  const radius = Math.max(1, (size - strokeWidth * 2 - 2) / 2)
  const center = size / 2
  const circumference = 2 * Math.PI * radius
  const indicator = (
    <span
      className="context-usage-indicator relative inline-flex min-w-0 items-center gap-1 text-slate-600 outline-none dark:text-slate-400"
      data-compacting={isCompacting}
      title={tooltip === "native" ? accessibleLabel : undefined}
      aria-label={accessibleLabel}
      tabIndex={tooltip === "popover" ? 0 : undefined}
      onMouseEnter={() => setActive(true)}
      onMouseLeave={() => setActive(false)}
      onFocus={() => setActive(true)}
      onBlur={() => setActive(false)}
    >
      <span
        aria-hidden="true"
        style={{
          width: size,
          height: size,
        }}
        className="box-border shrink-0 rounded-full bg-transparent"
      >
        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="block w-full h-full [transform:rotate(-90deg)]"
        >
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            strokeWidth={strokeWidth}
            className={
              active
                ? "stroke-slate-400 dark:stroke-slate-600"
                : "stroke-slate-300 dark:stroke-slate-750"
            }
          />
          {pct !== null && (
            <circle
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={`${circumference}`}
              strokeDashoffset={`${circumference * (1 - pct / 100)}`}
              className={`${contextUsageStrokeClass(pct, isCompacting)} ${
                isCompacting
                  ? "[transform-box:fill-box] [transform-origin:center] [animation:context-rewind_1.1s_linear_infinite] motion-reduce:[animation:none]"
                  : ""
              } transition-[stroke,stroke-dashoffset] duration-200`}
            />
          )}
        </svg>
      </span>

      {showTokenLabel && tokenEstimate !== null && tokenEstimate > 0 && (
        <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-[12px] text-slate-500 [font-variant-numeric:tabular-nums]">
          {formatTokensCompact(tokenEstimate)}
        </span>
      )}
    </span>
  )
  if (tooltip !== "popover") return indicator
  return (
    <Tooltip>
      <TooltipTrigger render={indicator} />
      <TooltipContent
        side="top"
        sideOffset={9}
        align={tooltipPlacement === "above-right" ? "end" : "center"}
        className="min-w-[190px] flex-col items-start gap-0 border border-slate-300 bg-white px-3 py-2.5 text-left text-slate-900 shadow-md dark:border-slate-600 dark:bg-slate-750 dark:text-slate-100"
      >
        <span className="block whitespace-nowrap text-[12px] font-semibold leading-4">
          {tooltipLines[0]}
        </span>
        <span className="mt-1 block whitespace-nowrap text-[11px] leading-4 text-slate-600 dark:text-slate-200">
          {tooltipLines[1]}
        </span>
        <span className="block whitespace-nowrap text-[11px] leading-4 text-slate-500 dark:text-slate-300">
          {tooltipLines[2]}
        </span>
      </TooltipContent>
    </Tooltip>
  )
}
