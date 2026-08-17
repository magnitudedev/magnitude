/**
 * Interrupted message — spec §9.3.9
 *
 * Standalone divider row.
 */
import { type ReactNode } from "react"
import { Square } from "lucide-react"
import type { InterruptedMessage as InterruptedType } from "@magnitudedev/sdk"
const DEFAULT_INTERRUPTED_TEXT =
  "Interrupted. What would you like to do instead?"
function getInterruptedText(message: InterruptedType): string {
  if (message.context === "root") {
    return DEFAULT_INTERRUPTED_TEXT
  }
  return "Agent interrupted."
}
export function InterruptedDivider({
  label = DEFAULT_INTERRUPTED_TEXT,
}: {
  label?: string
}): ReactNode {
  return (
    <div className="flex items-center [gap:10px] w-full [padding:18px_0_18px_12px] font-sans text-[12px] leading-[16px] text-slate-500">
      <span
        aria-hidden="true"
        className="h-px w-8 shrink-0 bg-slate-300 dark:bg-slate-750"
      />
      <span className="inline-flex items-center [gap:6px] [flex:0_1_auto] min-w-0">
        <Square size={10} fill="currentColor" className="shrink-0" />
        {label}
      </span>
      <span
        aria-hidden="true"
        className="h-px min-w-8 flex-1 bg-slate-300 dark:bg-slate-750"
      />
    </div>
  )
}
export function InterruptedMessage({
  message,
}: {
  message: InterruptedType
}): ReactNode {
  return <InterruptedDivider label={getInterruptedText(message)} />
}
