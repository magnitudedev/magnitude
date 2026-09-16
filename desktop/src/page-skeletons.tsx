import type { ReactNode } from "react"
import { CircleNotchIcon } from "@phosphor-icons/react"
import { RecommendationPreference } from "./model-preference-slider"
import { Skeleton } from "../../web/src/components/ui/skeleton"
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
  return <div className="my-3 aspect-[360/270] w-full" aria-hidden="true">
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
export function HardwarePending({ identifying = false }: { identifying?: boolean }) {
  return <div aria-busy="true" aria-label="Detecting your hardware" className={`relative ${pageLayout.hardware}`} data-loading-region="">
    <div className={`grid items-center gap-6 ${pageLayout.hardwareGrid}`}>
      <div aria-hidden="true" className={pageLayout.hardwarePhoto}><Skeleton className="aspect-[4/3] w-full rounded-none" /></div>
      <div className="min-w-0"><p className="text-xs font-medium uppercase tracking-widest text-slate-500">Your machine</p><LoadingStatus title={identifying ? "Identifying your machine" : "Reading hardware capabilities"} detail={identifying ? "Looking up your computer’s make and model." : "Checking your chip, graphics, and available memory."} />
        <div aria-hidden="true" className="mt-5 flex flex-wrap gap-x-7 gap-y-4">{[0,1].map(index => <div key={index} className="flex items-center gap-3"><Skeleton className="size-5 shrink-0" /><div><SkeletonLine className={index === 0 ? "h-7 w-24 text-lg" : "h-5 w-24 text-sm"} /><SkeletonLine className="h-4 w-24 text-xs" /></div></div>)}</div>
      </div>
    </div>
  </div>
}
export function RecommendationsSkeleton({ assessment, waitingForHardware = false }: { assessment?: { settledModels: number; totalModels: number }; waitingForHardware?: boolean }) {
  return <div aria-busy="true" aria-label="Loading recommendations" className="relative mb-8" data-loading-region="">
    <div className={pageLayout.recommendations}>
      <div aria-hidden="true" className={pageLayout.recommendationList}>{Array.from({length:5},(_,index) => <div key={index} className={`${pageLayout.recommendationRow} border-transparent`}><span className="w-4 shrink-0 text-sm text-slate-500">{index+1}</span><Skeleton className="size-7 shrink-0" /><SkeletonLine className="h-5 min-w-0 flex-1 text-sm" width={index%2 ? "90%" : "75%"} /></div>)}</div>
      <div className={pageLayout.recommendationPane}>
        <div aria-hidden="true" className={pageLayout.recommendationToolbar}><div className="flex gap-1"><Skeleton className="h-8 w-16" /><Skeleton className="h-8 w-16" /></div><Skeleton className="h-8 w-40" /></div>
        <div className="grid min-h-72">
          <div aria-hidden="true" className="invisible col-start-1 row-start-1 min-w-0"><RadarSkeleton /></div>
          <div className="col-start-1 row-start-1 flex min-w-0 items-center px-3">
          <LoadingStatus title={waitingForHardware ? "Waiting for hardware" : assessment ? "Assessing models" : "Loading model catalog"} detail={waitingForHardware ? "Recommendations need your machine’s capabilities." : assessment ? "Estimating memory fit and speed on your machine." : "Preparing model configurations for assessment."} assessment={waitingForHardware ? undefined : assessment} />
          </div>
        </div>
      </div>
    </div>
  </div>
}
export function ModelCardsSkeleton({ library = false }: { library?: boolean }) {
  return <LoadingRegion label={library ? "Loading your models" : "Loading catalog models"}>
    <div className="grid items-start gap-5">{[0,1,2,3].map(index => <article key={index} className={pageLayout.modelCard}><div className={pageLayout.modelRow}>
      <div className="flex min-w-0 items-center gap-4"><Skeleton className="size-10 shrink-0" /><div className="min-w-0 flex-1"><SkeletonLine className="h-7 text-lg" width={index%2 ? "85%" : "70%"} />{library && <SkeletonLine className="mt-1 h-5 text-sm" width="55%" />}</div></div>
      <div className="flex flex-wrap gap-2"><Skeleton className="h-8 w-20" /><Skeleton className={`h-8 ${library ? "w-28" : "w-40"}`} />{library && <Skeleton className="size-8" />}</div>
    </div></article>)}</div>
  </LoadingRegion>
}
export function ModelsSkeleton({ page }: { page: "discover" | "catalog" | "models" }) {
  if (page === "discover") return <><HardwarePending /><RecommendationPreference /><RecommendationsSkeleton /></>
  return <>
    <div className={pageLayout.modelHeader}><h1 className={pageLayout.pageTitle}>{page === "models" ? "My Models" : "Catalog"}</h1><Skeleton className="h-5 w-16" /></div>
    <div className={pageLayout.catalogToolbar}><div className="flex flex-wrap items-center gap-2"><Skeleton className="h-8 w-32" /><Skeleton className="h-8 w-44" /></div><Skeleton className={`h-8 ${pageLayout.modelSearch}`} /></div>
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
