/**
 * Thinking message — spec §9.3.4
 *
 * Default mode: hidden (handled by timeline grouping).
 * Transcript mode: collapsed disclosure with Brain icon.
 * Expanded: italic mono text with dashed left border.
 */
import { useState, type ReactNode } from "react"
import { Option } from "effect"
import { Brain, ChevronRight } from "lucide-react"
import type { ThinkingMessage as ThinkingMessageType } from "@magnitudedev/sdk"

export interface ThinkingMessageProps {
  message: ThinkingMessageType
  mode?: "default" | "transcript"
}

export function ThinkingMessage({
  message,
  mode = "transcript",
}: ThinkingMessageProps): ReactNode {
  const [expanded, setExpanded] = useState(false)
  const label = Option.getOrNull(message.label)

  if (mode === "default") return null

  return (
    <div style={{ padding: "4px 0" }}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="hover-fg-tertiary"
        style={{
          background: "transparent",
          border: "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: "4px",
          fontFamily: "var(--font-sans)",
          fontSize: "13px",
          padding: 0,
        }}
      >
        <Brain size={14} />
        <span>{label ?? "Thinking"}</span>
        <ChevronRight
          size={14}
          style={{
            transform: expanded ? "rotate(90deg)" : "none",
            transition: "transform 100ms ease",
          }}
        />
      </button>
      {expanded && (
        <div
          style={{
            marginTop: "4px",
            paddingLeft: "8px",
            borderLeft: "1px dashed var(--border-default)",
            fontFamily: "var(--font-mono)",
            fontSize: "13px",
            color: "var(--fg-secondary)",
            fontStyle: "italic",
            whiteSpace: "pre-wrap",
            lineHeight: 1.5,
          }}
        >
          {message.content}
        </div>
      )}
    </div>
  )
}
