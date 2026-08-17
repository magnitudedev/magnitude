/**
 * SidebarEmptyState & SidebarLoadingState — spec §10
 *
 * Empty state: MessageSquare icon, "No sessions found", helpful subtitle.
 * Loading state: skeleton rows sized to match real session rows.
 */
import { MessageSquare } from "lucide-react"
import type { ReactNode } from "react"

// ── Empty state ──

export function SidebarEmptyState({
  searchQuery,
}: {
  searchQuery?: string
}): ReactNode {
  return (
    <div className="flex flex-col items-center justify-center [padding:32px_16px] text-center">
      <MessageSquare size={24} className="text-slate-500 [margin-bottom:8px]" />
      <div className="font-sans text-[14px] text-slate-600 dark:text-slate-400">
        No sessions found
      </div>
      <div className="font-sans text-[12px] text-slate-500 [margin-top:4px]">
        {searchQuery
          ? "Try a different search"
          : "Create a new session to get started"}
      </div>
    </div>
  )
}

// ── Loading state (skeleton rows) ──

const SKELETON_COUNT = 5
export function SidebarLoadingState(): ReactNode {
  return (
    <>
      {Array.from(
        {
          length: SKELETON_COUNT,
        },
        (_, i) => (
          <div
            key={i}
            className="mb-1.5 flex cursor-pointer rounded px-2.5 py-2 bg-transparent transition-colors duration-100 hover:bg-slate-150 data-[selected=true]:bg-slate-200 dark:hover:bg-slate-800 dark:data-[selected=true]:bg-slate-750 cursor-default pointer-events-none"
            aria-hidden="true"
          >
            <div className="[flex:1] min-w-0 flex flex-col [gap:3px]">
              <div className="[height:22px] flex items-center [gap:8px] min-w-0">
                <div
                  style={{
                    width: `${i % 3 === 0 ? 68 : i % 3 === 1 ? 54 : 78}%`,
                  }}
                  className="[height:14px] rounded-[3px] bg-slate-100 dark:bg-slate-800 opacity-[0.56]"
                />
                <div className="[height:12px] [width:32px] rounded-[3px] bg-slate-100 dark:bg-slate-800 opacity-[0.38] shrink-0" />
              </div>
              <div className="[height:19px] flex items-center [gap:8px] min-w-0">
                <div
                  style={{
                    width: `${i % 2 === 0 ? 56 : 66}%`,
                  }}
                  className="[height:12px] rounded-[3px] bg-slate-100 dark:bg-slate-800 opacity-[0.34]"
                />
                <div className="[height:12px] [width:28px] rounded-[3px] bg-slate-100 dark:bg-slate-800 opacity-[0.28] shrink-0" />
              </div>
            </div>
          </div>
        )
      )}
    </>
  )
}
