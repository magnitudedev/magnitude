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

function usagePercent(context: ContextUsageDisplay | null, tokenCap: number | null | undefined): number | null {
  const tokenEstimate = context?.tokenEstimate ?? null
  if (tokenEstimate === null || !tokenCap || tokenCap <= 0) return null
  return Math.min(100, Math.max(0, (tokenEstimate / tokenCap) * 100))
}

function tooltipText(context: ContextUsageDisplay | null, tokenCap: number | null | undefined): string {
  const tokenEstimate = context?.tokenEstimate ?? null
  if (tokenEstimate === null) return "Context window unavailable"

  const tokens = formatTokensCompact(tokenEstimate)
  if (tokenCap && tokenCap > 0) {
    const pct = Math.round((tokenEstimate / tokenCap) * 100)
    return `Context window:\n${pct}% used (${100 - pct}% left)\n${tokens} / ${formatTokensCompact(tokenCap)} tokens used`
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
      className="context-usage-indicator"
      data-compacting={isCompacting}
      title={tooltip === "native" ? title : undefined}
      aria-label={title.replace(/\n/g, " ")}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: "relative",
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        minWidth: 0,
        color: "var(--fg-secondary)",
      }}
    >
      <span
        aria-hidden="true"
        style={{
          width: size,
          height: size,
          borderRadius: "50%",
          background: "transparent",
          boxSizing: "border-box",
          animation: isCompacting ? "context-pulse 900ms ease-in-out infinite" : "none",
          flexShrink: 0,
        }}
      >
        <svg
          viewBox={`0 0 ${size} ${size}`}
          style={{
            display: "block",
            width: "100%",
            height: "100%",
            transform: "rotate(-90deg)",
          }}
        >
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={hovered ? "var(--border-hover)" : "var(--border-default)"}
            strokeWidth={strokeWidth}
          />
          {pct !== null && (
            <circle
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              stroke="var(--accent-primary)"
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={`${circumference}`}
              strokeDashoffset={`${circumference * (1 - pct / 100)}`}
            />
          )}
        </svg>
      </span>

      {showTokenLabel && tokenEstimate !== null && tokenEstimate > 0 && (
        <span
          style={{
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            fontSize: 12,
            color: "var(--fg-tertiary)",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {formatTokensCompact(tokenEstimate)}
        </span>
      )}

      {popoverVisible && (
        <span
          role="tooltip"
          style={{
            position: "absolute",
            right: tooltipPlacement === "above-right" ? 0 : undefined,
            left: tooltipPlacement === "above-center" ? "50%" : undefined,
            transform: tooltipPlacement === "above-center" ? "translateX(-50%)" : undefined,
            bottom: "calc(100% + 8px)",
            minWidth: 170,
            padding: "8px 10px",
            borderRadius: 6,
            border: "1px solid var(--border-default)",
            background: "var(--bg-surface-elevated)",
            color: "var(--fg-primary)",
            fontSize: 12,
            lineHeight: 1.45,
            whiteSpace: "pre-line",
            textAlign: "center",
            zIndex: 30,
          }}
        >
          {title}
        </span>
      )}
    </span>
  )
}
