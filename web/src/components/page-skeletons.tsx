import type { ReactNode } from "react"
import { CircleNotchIcon } from "@phosphor-icons/react"
import { Skeleton } from "./ui/skeleton"
import { pageLayout } from "./page-layout"

export function LoadingRegion({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return <div aria-busy="true" aria-label={label} className={className} data-loading-region="">
    <span role="status" className="sr-only">{label}</span>
    <div aria-hidden="true">{children}</div>
  </div>
}

// A line occupies the same line box as text, with a shorter ink-shaped placeholder inside.
export function SkeletonLine({ className = "h-5", width = "70%" }: { className?: string; width?: string }) {
  return <span className={`flex max-w-full items-center ${className}`}><Skeleton className="h-[0.65em] max-w-full" style={{ width }} /></span>
}
export function HardwarePending({ identifying = false, startingService = false }: { identifying?: boolean; startingService?: boolean }) {
  return <div aria-busy="true" aria-label="Detecting your hardware" className="mb-5 flex items-center gap-3 py-1 text-xs text-slate-500" data-loading-region="">
    <CircleNotchIcon aria-hidden="true" className="size-4 motion-safe:animate-spin" />
    <span role="status">{startingService ? "Starting Magnitude…" : identifying ? "Identifying your machine…" : "Reading hardware capabilities…"}</span>
  </div>
}
export function ConnectionsSkeleton() {
  return <LoadingRegion label="Loading connections"><SkeletonLine className="mb-2 h-4 text-xs" width="140px" /><div className={pageLayout.harnessGrid}>{Array.from({length:8},(_,index) => <article className={pageLayout.harnessCard} key={index}><div className="flex flex-wrap items-center justify-between gap-3"><div className="flex min-w-0 items-center gap-3"><Skeleton className="size-8 shrink-0 rounded-lg" /><div><SkeletonLine className="h-5 w-24 text-sm" /><SkeletonLine className="mt-0.5 h-4 w-20 text-xs" /></div></div><Skeleton className="h-7 w-20" /></div></article>)}</div></LoadingRegion>
}
export function LoginSkeleton() {
  return <LoadingRegion label="Loading login settings" className={pageLayout.settingsCard}><div className="flex items-center justify-between gap-6"><div><h2 className="font-heading text-lg">Launch at login</h2><p className="mt-2 text-sm text-slate-500">Start Magnitude in the background with its tray icon. The window stays closed.</p></div><Skeleton className="h-8 w-16 shrink-0" /></div></LoadingRegion>
}
export function UpdatesSkeleton() {
  return <LoadingRegion label="Loading update settings" className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-750"><h3 className="font-medium">Application updates</h3><div className="mt-3 flex h-5 items-center gap-3"><Skeleton className="size-4" /><Skeleton className="h-3 w-40" /></div><SkeletonLine className="mt-3 h-5 text-sm" width="65%" /><Skeleton className="mt-3 h-8 w-32" /></LoadingRegion>
}
