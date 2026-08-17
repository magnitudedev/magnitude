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
    <div className="[padding:4px_0]">
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-slate-500 hover:text-slate-600 dark:hover:text-slate-400 [background:transparent] border-0 cursor-pointer flex items-center [gap:4px] font-sans text-[13px] [padding:0px]"
      >
        <Brain size={14} />
        <span>{label ?? "Thinking"}</span>
        <ChevronRight
          size={14}
          className={`${
            expanded ? "[transform:rotate(90deg)]" : "[transform:none]"
          }  [transition:transform_100ms_ease]`}
        />
      </button>
      {expanded && (
        <div className="mt-1 border-l border-dashed border-slate-300 pl-2 font-mono text-[13px] leading-[1.5] whitespace-pre-wrap text-slate-600 italic dark:border-slate-750 dark:text-slate-400">
          {message.content}
        </div>
      )}
    </div>
  )
}
