/**
 * FooterBar — spec §9.7
 *
 * Bottom dock status row matching the CLI's model-runtime footer.
 */
import { useState } from "react"
import { LoaderCircle } from "lucide-react"
import {
  formatCwdForDisplay,
  type ReasoningEffortOption,
} from "@magnitudedev/client-common"
import type { ContextUsageDisplay, ReasoningEffort } from "@magnitudedev/sdk"
import { formatFooterContextUsage } from "./local-inference-format"

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
  /** Current local-model residency presentation. */
  modelResidency?: "ready" | "loading" | "not-ready" | null
  /** Loading percentage, when projected by the selected slot. */
  modelLoadingPercentage?: number | null
  /** Thinking level label (e.g. "High", "Medium") */
  thinkingLevel?: string | null
  /** Resident model allocation label. */
  memoryLabel?: string | null
  /** Next Esc will kill all workers */
  nextEscWillKillAll?: boolean
  /** Transcript mode active */
  transcriptMode?: boolean
  /** Click handler for model name (opens the Models settings tab) */
  onModelClick?: () => void
  /** Click handler for resident memory (opens Hardware) */
  onMemoryClick?: () => void
  /** Reasoning efforts supported by the selected model. */
  thinkingOptions?: readonly ReasoningEffortOption[]
  /** Currently selected reasoning effort. */
  thinkingEffort?: ReasoningEffort | null
  /** Applies a reasoning effort to the primary model slot. */
  onThinkingSelect?: (effort: ReasoningEffort) => void
}

function ModelResidencyIndicator({
  residency,
  loadingPercentage,
}: {
  readonly residency: NonNullable<FooterBarProps["modelResidency"]>
  readonly loadingPercentage?: number | null
}): React.ReactNode {
  if (residency === "loading") {
    const label = loadingPercentage === null || loadingPercentage === undefined
      ? "Model loading"
      : `Model loading · ${loadingPercentage}%`
    return (
      <span className="footer-residency loading" aria-label={label} title={label}>
        <LoaderCircle className="spin" size={12} aria-hidden="true" />
      </span>
    )
  }
  const ready = residency === "ready"
  const label = ready ? "Model ready" : "Model not ready"
  return (
    <span
      className={`footer-residency ${ready ? "ready" : "not-ready"}`}
      aria-label={label}
      title={label}
    >
      <span aria-hidden="true">{ready ? "●" : "○"}</span>
    </span>
  )
}

export function FooterBar({
  context,
  tokenCap,
  bashMode,
  cwd,
  model,
  modelResidency,
  modelLoadingPercentage,
  thinkingLevel,
  memoryLabel,
  nextEscWillKillAll,
  transcriptMode,
  onModelClick,
  onMemoryClick,
  thinkingOptions = [],
  thinkingEffort,
  onThinkingSelect,
}: FooterBarProps): React.ReactNode {
  const [thinkingOpen, setThinkingOpen] = useState(false)
  const cwdText = cwd
    ? formatCwdForDisplay(cwd, { maxLen: 80, abbreviateHome: true })
    : ""
  const contextLabel = formatFooterContextUsage(context, tokenCap)

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
      {/* Model controls mirror the CLI footer: model and reasoning on the
          left, environment on the right. */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 8,
          minWidth: 0,
        }}
      >
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

        {!bashMode && modelResidency && (
          <ModelResidencyIndicator
            residency={modelResidency}
            loadingPercentage={modelLoadingPercentage}
          />
        )}
        {!bashMode && model && (
          <button
            type="button"
            onClick={onModelClick}
            disabled={!onModelClick}
            className="footer-action footer-model-action"
          >
            {model}
          </button>
        )}
        {!bashMode && thinkingLevel && (
          <button
            type="button"
            onClick={() => {
              if (thinkingOptions.length > 0 && onThinkingSelect) {
                setThinkingOpen((open) => !open)
              }
            }}
            aria-expanded={thinkingOptions.length > 0 ? thinkingOpen : undefined}
            aria-label={`Reasoning effort: ${thinkingLevel}`}
            disabled={thinkingOptions.length === 0 || !onThinkingSelect}
            className="footer-action footer-thinking-action"
          >
            {thinkingLevel}
          </button>
        )}
        {!bashMode && thinkingOpen && (
          <div
            role="listbox"
            aria-label="Reasoning effort"
            style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}
          >
            <span aria-hidden="true" style={{ color: "var(--fg-tertiary)" }}>&gt;</span>
            {thinkingOptions.map((option) => {
              const selected = option.value === thinkingEffort
              return (
                <button
                  key={option.value}
                  type="button"
                  role="option"
                  aria-selected={selected}
                  onClick={() => {
                    if (!selected) onThinkingSelect?.(option.value)
                    setThinkingOpen(false)
                  }}
                  style={{
                    padding: 0,
                    border: 0,
                    background: "transparent",
                    color: selected ? "var(--accent-violet)" : "var(--fg-secondary)",
                    font: "13px var(--font-sans)",
                    textDecoration: selected ? "underline" : "none",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                  }}
                >
                  {option.label}
                </button>
              )
            })}
          </div>
        )}
        {!bashMode && !thinkingOpen && (
          <>
            {memoryLabel && (
              <button
                type="button"
                onClick={onMemoryClick}
                disabled={!onMemoryClick}
                className="footer-action footer-memory-action"
              >
                {memoryLabel}
              </button>
            )}
            <span
              className="footer-context-usage"
              data-compacting={context?.isCompacting ?? false}
              title="Context usage"
            >
              {contextLabel}
            </span>
          </>
        )}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          minWidth: 0,
          maxWidth: "40%",
          marginLeft: 12,
        }}
      >
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
      </div>
    </div>
  )
}
