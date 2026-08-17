import { useState, type ReactNode } from "react"
import { formatTokensCompact } from "@magnitudedev/client-common"
import type { ContextUsageDisplay } from "@magnitudedev/sdk"
export interface ContextUsageIndicatorProps {
  context: ContextUsageDisplay | null
  tokenCap?: number | null
  size?: number
  strokeWidth?: number
  showTokenLabel?: boolean
  tooltip?: "popover" | "native" | "none"
  tooltipPlacement?: "above-right" | "above-center"
}
function usagePercent(
  context: ContextUsageDisplay | null,
  tokenCap: number | null | undefined
): number | null {
  const tokenEstimate = context?.tokenEstimate ?? null
  if (tokenEstimate === null || !tokenCap || tokenCap <= 0) return null
  return Math.min(100, Math.max(0, (tokenEstimate / tokenCap) * 100))
}
function tooltipText(
  context: ContextUsageDisplay | null,
  tokenCap: number | null | undefined
): string {
  const tokenEstimate = context?.tokenEstimate ?? null
  if (tokenEstimate === null) return "Context window unavailable"
  const tokens = formatTokensCompact(tokenEstimate)
  if (tokenCap && tokenCap > 0) {
    const pct = Math.round((tokenEstimate / tokenCap) * 100)
    return `Context window:\n${pct}% used (${
      100 - pct
    }% left)\n${tokens} / ${formatTokensCompact(tokenCap)} tokens used`
  }
  return `Context window:\n${tokens} tokens used`
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
  const [hovered, setHovered] = useState(false)
  const isCompacting = context?.isCompacting ?? false
  const tokenEstimate = context?.tokenEstimate ?? null
  const pct = usagePercent(context, tokenCap)
  const title = tooltipText(context, tokenCap)
  const radius = Math.max(1, (size - strokeWidth * 2 - 2) / 2)
  const center = size / 2
  const circumference = 2 * Math.PI * radius
  const popoverVisible = tooltip === "popover" && hovered
  return (
    <span
      className="context-usage-indicator relative inline-flex items-center [gap:4px] min-w-0 text-slate-600 dark:text-slate-400"
      data-compacting={isCompacting}
      title={tooltip === "native" ? title : undefined}
      aria-label={title.replace(/\n/g, " ")}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <span
        aria-hidden="true"
        style={{
          width: size,
          height: size,
        }}
        className={`${
          isCompacting
            ? "[animation:context-pulse_900ms_ease-in-out_infinite]"
            : "[animation:none]"
        }  rounded-full [background:transparent] box-border shrink-0`}
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
              hovered
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
              className="stroke-blue-700 dark:stroke-blue-500"
            />
          )}
        </svg>
      </span>

      {showTokenLabel && tokenEstimate !== null && tokenEstimate > 0 && (
        <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-[12px] text-slate-500 [font-variant-numeric:tabular-nums]">
          {formatTokensCompact(tokenEstimate)}
        </span>
      )}

      {popoverVisible && (
        <span
          role="tooltip"
          style={{
            right: tooltipPlacement === "above-right" ? 0 : undefined,
            left: tooltipPlacement === "above-center" ? "50%" : undefined,
            transform:
              tooltipPlacement === "above-center"
                ? "translateX(-50%)"
                : undefined,
          }}
          className="absolute [bottom:calc(100%_+_8px)] [min-width:170px] [padding:8px_10px] rounded-[6px] border border-slate-300 dark:border-slate-750 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-200 text-[12px] leading-[1.45] whitespace-pre-line text-center z-[30]"
        >
          {title}
        </span>
      )}
    </span>
  )
}
