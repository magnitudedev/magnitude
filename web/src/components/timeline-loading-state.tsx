/**
 * TimelineLoadingState — context-aware loading state for timelines.
 *
 * Shows a title, optional subtitle, and a spinner. No logo, no wordmark.
 * The title/subtitle provide context about what's loading (session title+cwd
 * for main, task+model for worker), following the TUI pattern where the
 * loading screen shows what's already known.
 */
import { Loader2 } from "lucide-react"
import type { ReactNode } from "react"

export interface TimelineLoadingStateProps {
  title: string
  subtitle?: string | null
}

export function TimelineLoadingState({
  title,
  subtitle,
}: TimelineLoadingStateProps): ReactNode {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        flex: 1,
        minHeight: 0,
        padding: "48px 24px",
        gap: 4,
      }}
    >
      {title && (
        <div
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: 14,
            color: "var(--fg-secondary)",
            textAlign: "center",
          }}
        >
          {title}
        </div>
      )}
      {subtitle && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--fg-tertiary)",
            textAlign: "center",
          }}
        >
          {subtitle}
        </div>
      )}
      <Loader2
        size={16}
        style={{
          color: "var(--accent-primary)",
          animation: "spin 1s linear infinite",
          marginTop: 16,
        }}
      />
    </div>
  )
}
