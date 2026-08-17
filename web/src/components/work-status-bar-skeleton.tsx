/**
 * WorkStatusBarSkeleton — placeholder mirroring the real WorkStatusBar
 * dimensions to prevent layout shift during session loading.
 *
 * Same outer box (border, radius, bg, minHeight 34px). Left: a 9px circle
 * + a 120px shimmer bar. Right: a 60px shimmer bar.
 */
import type { ReactNode } from "react"
export function WorkStatusBarSkeleton(): ReactNode {
  return (
    <div className="[margin:0px] border border-slate-300 dark:border-slate-750 rounded-[6px] bg-white dark:bg-slate-875 [min-height:34px] [padding:0_10px] flex items-center [gap:8px] shrink-0 overflow-hidden">
      {/* Left: status dot + status text bar */}
      <div className="[width:9px] [height:9px] rounded-full bg-slate-200 dark:bg-slate-800 shrink-0" />
      <div className="h-3 w-[120px] shrink-0 animate-shimmer rounded-[3px] bg-linear-to-r from-slate-200 via-white to-slate-200 bg-[length:200%_100%] dark:from-slate-800 dark:via-slate-875 dark:to-slate-800" />
      {/* Right: task count bar */}
      <div className="ml-auto h-3 w-[60px] shrink-0 animate-shimmer rounded-[3px] bg-linear-to-r from-slate-200 via-white to-slate-200 bg-[length:200%_100%] dark:from-slate-800 dark:via-slate-875 dark:to-slate-800" />
    </div>
  )
}
