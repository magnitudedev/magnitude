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
export function ModelCardsSkeleton({ library = false }: { library?: boolean }) {
  return <LoadingRegion label={library ? "Loading your models" : "Loading catalog models"}>
    <div className="grid items-start gap-3">{[0,1,2,3].map(index => <article key={index} className="overflow-hidden rounded-2xl border border-slate-200 bg-white dark:border-slate-750 dark:bg-slate-850">
      <div className="flex items-center gap-4 px-5 py-4"><Skeleton className="size-10 shrink-0" /><div className="min-w-0 flex-1"><SkeletonLine className="h-6 text-[15px]" width={index%2 ? "55%" : "40%"} /><SkeletonLine className="h-4 text-xs" width={library ? "30%" : "22%"} /></div><Skeleton className="h-8 w-36" /></div>
      <div className={`grid gap-px border-t border-slate-200 bg-slate-200 dark:border-slate-750 dark:bg-slate-750 ${library ? "grid-cols-3 lg:grid-cols-5" : "grid-cols-2 lg:grid-cols-4"}`}>{Array.from({ length: library ? 5 : 4 }, (_, cell) => <div key={cell} className="bg-white px-4 py-2.5 dark:bg-slate-850"><SkeletonLine className="h-6 text-[13px]" width="65%" /></div>)}</div>
    </article>)}</div>
  </LoadingRegion>
}
export function ModelsSkeleton({ startingService = false }: { startingService?: boolean }) {
  return <>
    <div className={pageLayout.modelHeader}><h1 className={pageLayout.pageTitle}>Models</h1><Skeleton className="h-8 w-36" /></div>
    {startingService && <HardwarePending startingService />}
    <div className={pageLayout.catalogToolbar}><Skeleton className={`h-8 ${pageLayout.modelSearch}`} /><div className="flex flex-wrap items-center gap-2"><Skeleton className="h-8 w-32" /><Skeleton className="h-8 w-28" /><Skeleton className="h-8 w-36" /></div></div>
    <ModelCardsSkeleton library />
  </>
}
export function ConnectionsSkeleton() {
  return <LoadingRegion label="Loading connections" className="mt-7 space-y-8"><SkeletonLine className="mb-4 h-5 text-sm" width="180px" /><div className={pageLayout.harnessGrid}>{Array.from({length:8},(_,index) => <article className={pageLayout.harnessCard} key={index}><div className="flex flex-wrap items-center justify-between gap-4"><div className="flex min-w-0 items-center gap-3"><Skeleton className="size-14 shrink-0 rounded-2xl" /><div><SkeletonLine className="h-7 w-32 text-lg" /><SkeletonLine className="mt-1 h-5 w-28 text-sm" /></div></div><Skeleton className="h-8 w-24" /></div></article>)}</div></LoadingRegion>
}
export function LoginSkeleton() {
  return <LoadingRegion label="Loading login settings" className={pageLayout.settingsCard}><div className="flex items-center justify-between gap-6"><div><h2 className="font-heading text-lg">Launch at login</h2><p className="mt-2 text-sm text-slate-500">Start Magnitude in the background with its tray icon. The window stays closed.</p></div><Skeleton className="h-8 w-16 shrink-0" /></div></LoadingRegion>
}
export function UpdatesSkeleton() {
  return <LoadingRegion label="Loading update settings" className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-750"><h3 className="font-medium">Application updates</h3><div className="mt-3 flex h-5 items-center gap-3"><Skeleton className="size-4" /><Skeleton className="h-3 w-40" /></div><SkeletonLine className="mt-3 h-5 text-sm" width="65%" /><Skeleton className="mt-3 h-8 w-32" /></LoadingRegion>
}
