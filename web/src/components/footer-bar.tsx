/**
 * FooterBar — spec §9.7
 *
 * Bottom dock status row: cwd, model, and radial context usage.
 */
import { formatCwdForDisplay } from "@magnitudedev/client-common"
import type { ContextUsageDisplay } from "@magnitudedev/sdk"
import { ContextUsageIndicator } from "./context-usage-indicator"

export interface FooterBarProps {
  /** Context usage info from timeline */
  context: ContextUsageDisplay | null
  /** Token cap (max context window tokens), if known */
  tokenCap?: number | null
  /** Bash mode active */
  bashMode?: boolean
  /** Current agent-host working directory */
  cwd?: string | null
  /** Current model label */
  model?: string | null
  /** Thinking level label (e.g. "High", "Medium") */
  thinkingLevel?: string | null
  /** Next Esc will kill all workers */
  nextEscWillKillAll?: boolean
  /** Transcript mode active */
  transcriptMode?: boolean
  /** Click handler for model name (opens settings) */
  onModelClick?: () => void
  /** Click handler for thinking level (opens settings) */
  onThinkingClick?: () => void
}

export function FooterBar({
  context,
  tokenCap,
  bashMode,
  cwd,
  model,
  thinkingLevel,
  nextEscWillKillAll,
  transcriptMode,
  onModelClick,
  onThinkingClick,
}: FooterBarProps): React.ReactNode {
  const cwdText = cwd
    ? formatCwdForDisplay(cwd, { maxLen: 80, abbreviateHome: true })
    : ""

  return (
    <div
      className="footer-bar"
      style={{
        minHeight: 26,
        padding: "0 2px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexShrink: 0,
        background: "transparent",
        fontFamily: "var(--font-sans)",
      }}
    >
      {/* Left side */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        {cwdText && (
          <span
            title={cwd ?? undefined}
            style={{
              color: "var(--fg-secondary)",
              fontFamily: "var(--font-mono)",
              fontSize: 13,
              minWidth: 0,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {cwdText}
          </span>
        )}
        {bashMode && (
          <span style={{ fontSize: 11, color: "var(--accent-warning)", flexShrink: 0 }}>
            Bash mode
          </span>
        )}
        {nextEscWillKillAll && (
          <span style={{ fontSize: 11, color: "var(--accent-warning)", flexShrink: 0 }}>
            Press Esc again to interrupt all workers
          </span>
        )}
      </div>

      {/* Right side */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {transcriptMode && (
          <span
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 11,
              color: "var(--accent-info)",
              background: "var(--bg-surface-elevated)",
              border: "1px solid var(--border-default)",
              borderRadius: 4,
              padding: "1px 6px",
            }}
          >
            Transcript Mode
          </span>
        )}

        {model && (
          <span
            onClick={onModelClick}
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              color: "var(--fg-secondary)",
              cursor: onModelClick ? "pointer" : "default",
              textDecoration: onModelClick ? "underline" : "none",
            }}
          >
            {model}
          </span>
        )}
        {thinkingLevel && (
          <span
            onClick={onThinkingClick}
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              color: "var(--fg-secondary)",
              cursor: onThinkingClick ? "pointer" : "default",
              textDecoration: onThinkingClick ? "underline" : "none",
            }}
          >
            {thinkingLevel}
          </span>
        )}

        <ContextUsageIndicator context={context} tokenCap={tokenCap} />
      </div>
    </div>
  )
}
