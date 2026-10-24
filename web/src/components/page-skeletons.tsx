import type { ReactNode } from "react"
import { CircleNotchIcon } from "@phosphor-icons/react"
import { RecommendationPreference } from "./model-preference-slider"
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
export function RadarSkeleton() {
  return <div className={pageLayout.modelRadar} aria-hidden="true">
    <svg viewBox="0 0 360 270" className="h-full w-full motion-safe:animate-pulse fill-slate-200 dark:fill-slate-750">
      <polygon points="180,58 256,113 227,203 133,203 104,113" fill="none" className="stroke-slate-200 dark:stroke-slate-750" strokeWidth="2" />
      {[[180,20],[290,83],[258,238],[102,238],[70,83]].map(([x,y]) => <g key={`${x}-${y}`}><rect x={x!-30} y={y!-8} width="60" height="8" rx="3" /><rect x={x!-40} y={y!+9} width="80" height="11" rx="3" /></g>)}
    </svg>
  </div>
}
function LoadingStatus({ title, detail, assessment }: { title: string; detail: string; assessment?: { settledModels: number; totalModels: number } }) {
  return <div className="w-full max-w-sm py-3">
    <div role="status" className="text-sm">
      <div className="flex items-center gap-2"><CircleNotchIcon aria-hidden="true" className="size-4 shrink-0 text-blue-500 motion-safe:animate-spin" /><p className="font-medium">{title}</p></div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">{detail}</p>
      {assessment && assessment.totalModels > 0 && <>
        <progress aria-label="Model assessments completed" max={assessment.totalModels} value={assessment.settledModels} className="mt-3 block h-1.5 w-full overflow-hidden rounded-full [&::-webkit-progress-bar]:bg-slate-200 dark:[&::-webkit-progress-bar]:bg-slate-700 [&::-webkit-progress-value]:bg-blue-500 [&::-moz-progress-bar]:bg-blue-500" />
        <p className="mt-2 text-xs tabular-nums text-slate-500 dark:text-slate-400">{assessment.settledModels} of {assessment.totalModels} assessed</p>
      </>}
    </div>
  </div>
}
export function HardwarePending({ identifying = false, startingService = false }: { identifying?: boolean; startingService?: boolean }) {
  return <div aria-busy="true" aria-label="Detecting your hardware" className="mb-6 flex items-center gap-3 py-1 text-xs text-slate-500" data-loading-region="">
    <CircleNotchIcon aria-hidden="true" className="size-4 motion-safe:animate-spin" />
    <span role="status">{startingService ? "Starting Magnitude…" : identifying ? "Identifying your machine…" : "Reading hardware capabilities…"}</span>
  </div>
}
export function RecommendationsSkeleton({ assessment, waitingForHardware = false, waitingForService = false }: { assessment?: { settledModels: number; totalModels: number }; waitingForHardware?: boolean; waitingForService?: boolean }) {
  return <div aria-busy="true" aria-label="Loading recommendations" className="relative mb-8" data-loading-region="">
    <div className={pageLayout.recommendations}>
      <div aria-hidden="true" className={pageLayout.recommendationList}>{Array.from({length:5},(_,index) => <div key={index} className={`${pageLayout.recommendationRow} border-transparent`}><span className="w-4 shrink-0 text-sm text-slate-500">{index+1}</span><Skeleton className="size-7 shrink-0" /><SkeletonLine className="h-5 min-w-0 flex-1 text-sm" width={index%2 ? "90%" : "75%"} /></div>)}</div>
      <div className={pageLayout.recommendationPane}>
        <div aria-hidden="true" className="mb-4 flex justify-end"><Skeleton className="h-8 w-40" /></div>
        <div className="grid min-h-72">
          <div aria-hidden="true" className="invisible col-start-1 row-start-1 min-w-0"><RadarSkeleton /></div>
          <div className="col-start-1 row-start-1 flex min-w-0 items-center px-3">
          <LoadingStatus title={waitingForService ? "Waiting for the service" : waitingForHardware ? "Waiting for hardware" : assessment ? assessment.totalModels > 0 ? "Assessing models" : "Preparing models for assessment" : "Loading model catalog"} detail={waitingForService ? "Hardware detection and model assessment begin when it is ready." : waitingForHardware ? "Recommendations need your machine’s capabilities." : assessment ? "Estimating memory fit and speed on your machine." : "Preparing model configurations for assessment."} assessment={waitingForHardware || waitingForService ? undefined : assessment} />
          </div>
        </div>
      </div>
    </div>
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
export function ModelsSkeleton({ page, startingService = false }: { page: "discover" | "catalog" | "models"; startingService?: boolean }) {
  if (page !== "models") return <>
    <div className="mb-6 flex items-center justify-between gap-3"><h1 className={pageLayout.pageTitle}>Discover</h1><Skeleton className="h-8 w-48" /></div>
    <HardwarePending startingService={startingService} />
    <div aria-hidden="true" className="mb-6 flex flex-wrap items-center justify-between gap-6 border-b border-slate-200 pb-2 dark:border-slate-750"><div className="flex gap-6 py-2 text-sm text-slate-500"><span>Recommended</span><span>All models</span></div><Skeleton className="h-8 w-64" /></div>
    {page === "discover" ? <><RecommendationPreference /><RecommendationsSkeleton waitingForService={startingService} /></> : <ModelCardsSkeleton />}
  </>
  return <>
    <div className={pageLayout.modelHeader}><h1 className={pageLayout.pageTitle}>{page === "models" ? "My Models" : "Catalog"}</h1><Skeleton className={`h-8 ${page === "models" ? "w-36" : "w-48"}`} /></div>
    <div className={pageLayout.catalogToolbar}><Skeleton className={`h-8 ${pageLayout.modelSearch}`} /><div className="flex flex-wrap items-center gap-2"><Skeleton className="h-8 w-32" /><Skeleton className="h-8 w-28" /><Skeleton className="h-8 w-36" /></div></div>
    <ModelCardsSkeleton library={page === "models"} />
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
