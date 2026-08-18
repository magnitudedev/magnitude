import { Button } from "@/components/ui/button"

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
export function AgentCommunication({
  message,
}: {
  message: AgentCommType
}): ReactNode {
  const [expanded, setExpanded] = useState(false)
  const agentRole = Option.getOrNull(message.agentRole)
  const agentName = Option.getOrNull(message.agentName)
  const status = Option.getOrNull(message.status)
  const roleLabel = agentRole ? capitalize(agentRole) : agentName ?? "Agent"
  const directionLabel =
    message.direction === "from_agent" ? (
      <>
        Lead <span className="text-slate-500">→</span> {roleLabel}
      </>
    ) : (
      <>
        {roleLabel} <span className="text-slate-500">→</span> Lead
      </>
    )
  const lineCount = message.content.split("\n").length
  const canExpand = lineCount > COLLAPSED_LINE_LIMIT
  const displayContent =
    canExpand && !expanded
      ? message.content.split("\n").slice(0, COLLAPSED_LINE_LIMIT).join("\n") +
        "..."
      : message.content
  return (
    <div className="[padding-left:12px]">
      <div className="flex items-center [gap:6px] [padding:2px_0]">
        <Mail
          size={14}
          className="text-slate-600 dark:text-slate-400 shrink-0"
        />
        <span className="font-sans text-[13px]">
          <span className="text-blue-700 dark:text-blue-500 font-semibold">
            {directionLabel}
          </span>
        </span>
      </div>
      <div
        className={`${
          expanded ? "overflow-auto" : "overflow-hidden"
        }  [margin-top:2px] text-slate-600 dark:text-slate-400 text-[13px]`}
      >
        <MarkdownContent
          content={displayContent}
          isStreaming={status === "streaming"}
          showCursor={status === "streaming"}
          className="text-[13px] text-slate-600 dark:text-slate-400"
        />
      </div>
      {canExpand && (
        <Button variant="unstyled" size="unstyled"
          onClick={() => setExpanded(!expanded)}
          className="[background:transparent] border-0 cursor-pointer flex items-center [gap:4px] text-slate-500 font-sans text-[11px] [padding:0px] [margin-top:2px]"
        >
          <ChevronDown
            size={12}
            className={`${
              expanded ? "[transform:rotate(180deg)]" : "[transform:none]"
            }  [transition:transform_100ms_ease]`}
          />
          {expanded ? "Show less" : "Show more"}
        </Button>
      )}
    </div>
  )
}
