import type { ReactNode } from "react"
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
export function HardwareSkeleton() {
  return <LoadingRegion label="Loading your hardware" className={pageLayout.hardware}>
    <div className={`grid items-center gap-6 ${pageLayout.hardwareGrid}`}>
      <div className={pageLayout.hardwarePhoto}><Skeleton className="aspect-[4/3] w-full rounded-none" /></div>
      <div className="min-w-0"><p className="text-xs font-medium uppercase tracking-widest text-slate-500">Your machine</p><SkeletonLine className="mt-2 h-7 text-xl" width="80%" /><SkeletonLine className="mt-2 h-5 text-sm" width="55%" />
        <div className="mt-5 flex flex-wrap gap-x-7 gap-y-4">{[0,1].map(index => <div key={index} className="flex items-center gap-3"><Skeleton className="size-5 shrink-0" /><div><SkeletonLine className={index === 0 ? "h-7 w-24 text-lg" : "h-5 w-24 text-sm"} /><SkeletonLine className="h-4 w-24 text-xs" /></div></div>)}</div>
      </div>
    </div>
  </LoadingRegion>
}
export function RecommendationsSkeleton() {
  return <LoadingRegion label="Loading recommendations" className="mb-8">
    <div className={pageLayout.recommendations}>
      <div className={pageLayout.recommendationList}>{Array.from({length:5},(_,index) => <div key={index} className={`${pageLayout.recommendationRow} border-transparent`}><span className="w-4 shrink-0 text-sm text-slate-500">{index+1}</span><Skeleton className="size-7 shrink-0" /><SkeletonLine className="h-5 min-w-0 flex-1 text-sm" width={index%2 ? "90%" : "75%"} /></div>)}</div>
      <div className={pageLayout.recommendationPane}><div className={pageLayout.recommendationToolbar}><div className="flex gap-1"><Skeleton className="h-8 w-16" /><Skeleton className="h-8 w-16" /></div><Skeleton className="h-8 w-40" /></div><div className="grid min-h-72"><RadarSkeleton /></div></div>
    </div>
  </LoadingRegion>
}
export function ModelCardsSkeleton({ library = false }: { library?: boolean }) {
  return <LoadingRegion label={library ? "Loading your models" : "Loading catalog models"}>
    <div className="grid items-start gap-5">{[0,1,2,3].map(index => <article key={index} className={pageLayout.modelCard}><div className={pageLayout.modelRow}>
      <div className="flex min-w-0 items-center gap-4"><Skeleton className="size-10 shrink-0" /><div className="min-w-0 flex-1"><SkeletonLine className="h-7 text-lg" width={index%2 ? "85%" : "70%"} />{library && <SkeletonLine className="mt-1 h-5 text-sm" width="55%" />}</div></div>
      <div className="flex flex-wrap gap-2"><Skeleton className="h-8 w-20" /><Skeleton className={`h-8 ${library ? "w-28" : "w-40"}`} />{library && <Skeleton className="size-8" />}</div>
    </div></article>)}</div>
  </LoadingRegion>
}
export const modelDescriptions = { discover: "Your best local models, matched to your machine.", catalog: "Explore every model in the curated catalog.", models: "Your downloads and installed models, in one place." } as const
export function ModelsSkeleton({ page }: { page: keyof typeof modelDescriptions }) {
  return <><p className="mt-2 text-slate-500">{modelDescriptions[page]}</p>{page === "discover" ? <><HardwareSkeleton /><LoadingRegion label="Loading recommendation preference" className="my-6"><div className="flex items-end justify-between gap-4"><div><h2 className="font-heading text-xl">Find your balance</h2><p className="mt-2 text-sm text-slate-500">Quick responses or deeper thinking. Choose what matters to you.</p></div><Skeleton className="h-9 w-24 shrink-0 rounded-full" /></div><div className="mt-5"><div className="flex h-6 items-center"><Skeleton className="h-1.5 w-full" /></div><div className="mt-2 flex h-8 items-center justify-between">{[0,1,2,3,4].map(index => <Skeleton key={index} className="h-2 w-12" />)}</div></div></LoadingRegion><RecommendationsSkeleton /></> : <><div className={pageLayout.catalogToolbar}><h2 className="font-heading text-xl">{page === "models" ? "Your library" : "Explore the catalog"}</h2><Skeleton className="h-8 w-full max-w-sm" /></div>{page === "catalog" && <div className="mb-5 flex h-5 items-center justify-between"><Skeleton className="h-3 w-32" /><Skeleton className="h-3 w-28" /></div>}<ModelCardsSkeleton library={page === "models"} /></>}</>
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
