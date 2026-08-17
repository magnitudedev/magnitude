/**
 * WorkerDetailPanel — fills the active chat-column page with a worker timeline
 * as a read-only view. The page chrome is provided by ChatColumnPage.
 */
import type { ReactNode } from "react"
import { ChatTimeline } from "./chat-timeline"
export interface WorkerDetailPanelProps {
  forkId: string | null
  worker: {
    forkId: string
    role: string
    name: string
  } | null
  loadingTitle?: string
  loadingSubtitle?: string | null
}
export function WorkerDetailPanel({
  forkId,
  worker,
  loadingTitle,
  loadingSubtitle,
}: WorkerDetailPanelProps): ReactNode {
  if (!forkId) return null
  return (
    <div className="worker-detail-panel [flex:1] min-h-0 flex flex-col overflow-hidden bg-slate-50 dark:bg-slate-875">
      <ChatTimeline
        forkId={forkId}
        loadingTitle={loadingTitle}
        loadingSubtitle={loadingSubtitle}
      />
    </div>
  )
}
