/**
 * Agent communication message — spec §9.3.13
 *
 * Mail icon + directional label. "Lead → {Role}" or "{Role} → Lead".
 * Content as MarkdownContent, truncated to 6 lines with expand/collapse.
 */
import { useState, type ReactNode } from "react"
import { Option } from "effect"
import { Mail, ChevronDown } from "lucide-react"
import type { AgentCommunicationMessage as AgentCommType } from "@magnitudedev/sdk"
import { MarkdownContent } from "../markdown-content"

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

const COLLAPSED_LINE_LIMIT = 6

export function AgentCommunication({ message }: { message: AgentCommType }): ReactNode {
  const [expanded, setExpanded] = useState(false)
  const agentRole = Option.getOrNull(message.agentRole)
  const agentName = Option.getOrNull(message.agentName)
  const status = Option.getOrNull(message.status)
  const roleLabel = agentRole ? capitalize(agentRole) : (agentName ?? "Agent")
  const directionLabel = message.direction === "from_agent"
    ? <>Lead <span style={{ color: "var(--fg-tertiary)" }}>→</span> {roleLabel}</>
    : <>{roleLabel} <span style={{ color: "var(--fg-tertiary)" }}>→</span> Lead</>

  const lineCount = message.content.split("\n").length
  const canExpand = lineCount > COLLAPSED_LINE_LIMIT
  const displayContent = canExpand && !expanded
    ? message.content.split("\n").slice(0, COLLAPSED_LINE_LIMIT).join("\n") + "..."
    : message.content

  return (
    <div style={{ paddingLeft: "12px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "6px", padding: "2px 0" }}>
        <Mail size={14} style={{ color: "var(--fg-secondary)", flexShrink: 0 }} />
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "13px" }}>
          <span style={{ color: "var(--accent-primary)", fontWeight: 600 }}>{directionLabel}</span>
        </span>
      </div>
      <div
        style={{
          marginTop: "2px",
          color: "var(--fg-secondary)",
          fontSize: "13px",
          overflow: expanded ? "auto" : "hidden",
        }}
      >
        <MarkdownContent
          content={displayContent}
          isStreaming={status === "streaming"}
          showCursor={status === "streaming"}
          style={{ fontSize: "13px", color: "var(--fg-secondary)" }}
        />
      </div>
      {canExpand && (
        <button
          onClick={() => setExpanded(!expanded)}
          style={{
            background: "transparent",
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "4px",
            color: "var(--fg-tertiary)",
            fontFamily: "var(--font-sans)",
            fontSize: "11px",
            padding: 0,
            marginTop: "2px",
          }}
        >
          <ChevronDown
            size={12}
            style={{ transform: expanded ? "rotate(180deg)" : "none", transition: "transform 100ms ease" }}
          />
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  )
}
