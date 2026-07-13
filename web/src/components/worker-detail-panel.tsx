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
    <div
      className="worker-detail-panel"
      style={{
        flex: 1,
        minHeight: 0,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        background: "var(--bg-base)",
      }}
    >
      <ChatTimeline
        forkId={forkId}
        loadingTitle={loadingTitle}
        loadingSubtitle={loadingSubtitle}
      />
    </div>
  )
}
