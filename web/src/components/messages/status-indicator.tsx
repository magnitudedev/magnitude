/**
 * Status indicator message — spec §9.3.6
 *
 * Plain text, --fg-secondary, font-mono. No background, border, or icon.
 */
import { type ReactNode } from "react"
import type { StatusIndicatorMessage as StatusIndicatorType } from "@magnitudedev/sdk"
export function StatusIndicator({
  message,
}: {
  message: StatusIndicatorType
}): ReactNode {
  return (
    <div className="text-slate-600 dark:text-slate-400 text-[13px] font-mono">
      {message.message}
    </div>
  )
}
