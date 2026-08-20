import { useState, useSyncExternalStore, type ReactNode } from "react"
import { Option } from "effect"
import { ChevronRight } from "lucide-react"
import type { ThinkingMessage as ThinkingMessageType } from "@magnitudedev/sdk"
import {
  formatWorkDuration,
  getTickSnapshot,
  subscribeNoop,
  subscribeTick,
} from "@magnitudedev/client-common"
import { Button } from "@/components/ui/button"

const activeTextClass =
  "animate-shimmer bg-linear-to-r from-slate-700 via-slate-400 to-slate-700 bg-[length:200%_100%] bg-clip-text text-transparent motion-reduce:animate-none motion-reduce:[background:none] motion-reduce:text-slate-800 dark:from-slate-400 dark:via-white dark:to-slate-400 dark:motion-reduce:text-slate-200"

export interface ThinkingMessageProps {
  readonly message: ThinkingMessageType
}

export function ThinkingMessage({ message }: ThinkingMessageProps): ReactNode {
  const [expanded, setExpanded] = useState(false)
  const active = message.phase === "active"
  const tick = useSyncExternalStore(
    active ? subscribeTick : subscribeNoop,
    getTickSnapshot,
    getTickSnapshot,
  )
  const completedAt = Option.getOrElse(message.completedAt, () => tick)
  const duration = formatWorkDuration(Math.max(0, completedAt - message.timestamp))
  const label = Option.getOrNull(message.label)

  return (
    <div className="py-1 font-sans">
      <Button
        variant="unstyled"
        size="unstyled"
        type="button"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
        className="flex min-h-7 items-center gap-1.5 rounded-md border-0 bg-transparent px-1.5 text-[13px] text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-850"
      >
        <ChevronRight
          size={13}
          className={`${expanded ? "rotate-90" : "rotate-0"} shrink-0 transition-transform duration-100`}
        />
        <span className={active ? activeTextClass : "text-slate-700 dark:text-slate-300"}>
          {active ? (label ?? "Thinking") : `Thought for ${duration}`}
        </span>
        {active ? (
          <span className="ml-1 tabular-nums text-[12px] text-slate-500">
            {duration}
          </span>
        ) : null}
      </Button>
      {expanded ? (
        <div className="ml-2 mt-1.5 border-l border-slate-300 pl-3 font-sans text-[13px] leading-5 whitespace-pre-wrap text-slate-600 dark:border-slate-750 dark:text-slate-400">
          {message.content}
        </div>
      ) : null}
    </div>
  )
}
