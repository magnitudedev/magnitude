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
    <div className="flex flex-col items-center justify-center [flex:1] min-h-0 [padding:48px_24px] [gap:4px]">
      {title && (
        <div className="font-sans text-[14px] text-slate-600 dark:text-slate-400 text-center">
          {title}
        </div>
      )}
      {subtitle && (
        <div className="font-mono text-[12px] text-slate-500 text-center">
          {subtitle}
        </div>
      )}
      <Loader2
        size={16}
        className="text-blue-700 dark:text-blue-500 [animation:spin_1s_linear_infinite] [margin-top:16px]"
      />
    </div>
  )
}
