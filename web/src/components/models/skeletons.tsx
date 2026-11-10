import { Skeleton } from "../ui/skeleton"
import { HardwarePending, LoadingRegion, SkeletonLine } from "../page-skeletons"
import { pageLayout } from "../page-layout"
import { cn } from "@/lib/utils"
import { cardClass, Section } from "./parts"

/** The machine card before the hardware report: its photo, name and two figures. */
export function MachineCardSkeleton() {
  return <LoadingRegion label="Reading your hardware" className={cn(cardClass, "flex flex-wrap items-center gap-x-4 gap-y-4 px-5 py-4")}>
    <div className="flex flex-wrap items-center gap-x-4 gap-y-4">
      <Skeleton className="aspect-[4/3] w-16 shrink-0" />
      <div className="min-w-56 flex-1">
        <h2 className="font-heading text-base font-semibold">Your machine</h2>
        <SkeletonLine className="h-[21px] text-[14px]" width="12rem" />
      </div>
    </div>
    <div className="ml-auto grid shrink-0 grid-cols-2 items-start gap-8">
      {[0, 1].map(index => <div key={index}>
        <SkeletonLine className="h-4 text-xs" width="4rem" />
        <SkeletonLine className="mt-0.5 h-7 text-xl" width="5rem" />
      </div>)}
    </div>
  </LoadingRegion>
}

/** Collapsed offered models: lab mark, name and quant line, speed and size, and the selection mark. */
export function OfferedModelsSkeleton({ count = 3 }: { count?: number }) {
  return <LoadingRegion label="Choosing models for your machine" className="space-y-2">
    {Array.from({ length: count }, (_, index) => <div key={index} className={cn(cardClass, "grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3")}>
      <Skeleton className="size-10" />
      <span className="min-w-0">
        <SkeletonLine className="h-[21px] text-[14px]" width={index % 2 ? "9rem" : "12rem"} />
        <SkeletonLine className="mt-0.5 h-4 text-xs" width="60%" />
      </span>
      <span className="flex items-center gap-4">
        <span className="grid grid-cols-[84px_64px] gap-4">
          {[0, 1].map(cell => <span key={cell} className="flex w-full flex-col items-end">
            <SkeletonLine className="h-[22px] justify-end text-[15px]" width={cell ? "3.5rem" : "4.5rem"} />
            <SkeletonLine className="h-[16px] justify-end text-[11px]" width={cell ? "2.5rem" : "4rem"} />
          </span>)}
        </span>
        <Skeleton className="size-5 rounded-full" />
      </span>
    </div>)}
  </LoadingRegion>
}

/** Recommendations while the service is still measuring: how far along it is, then the rows about to appear. */
export function MeasuringRecommendations({ settled, total }: { settled: number; total: number }) {
  return <div>
    <div role="status" aria-live="polite" className="mb-3">
      <p className="flex items-center justify-between gap-3 text-[13px] text-slate-600 dark:text-slate-400">
        <span>Measuring how each model runs on your machine…</span>
        {total > 0 && <span className="text-xs tabular-nums text-slate-500">{settled} of {total} models</span>}
      </p>
      <div className="mt-2 h-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-750">
        <div className="h-full rounded-full bg-blue-600 transition-[width] duration-500 dark:bg-blue-400" style={{ width: `${total > 0 ? Math.max(0.04, settled / total) * 100 : 4}%` }} />
      </div>
    </div>
    <OfferedModelsSkeleton />
  </div>
}

/** The user's models list before it loads: title, find field and filter, and rows in the list's columns. */
export function YourModelsSkeleton({ count = 4 }: { count?: number }) {
  return <Section title="Your models" aside={<><Skeleton className="h-7 w-52" /><Skeleton className="size-7" /></>}>
    <LoadingRegion label="Loading your models" className="divide-y divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white dark:divide-slate-750 dark:border-slate-750 dark:bg-slate-850">
      {Array.from({ length: count }, (_, index) => <div key={index} className="grid grid-cols-[28px_minmax(0,1fr)_64px_72px_auto_28px] items-center gap-4 px-4 py-3 max-lg:grid-cols-[28px_minmax(0,1fr)_64px_auto_28px]">
        <Skeleton className="size-7" />
        <span className="min-w-0">
          <SkeletonLine className="h-5 text-[13px]" width={index % 2 ? "35%" : "50%"} />
          <SkeletonLine className="h-4 text-xs" width="30%" />
        </span>
        <SkeletonLine className="h-5 justify-end text-[13px]" width="3.5rem" />
        <SkeletonLine className="h-4 justify-end text-xs max-lg:hidden" width="3.5rem" />
        <Skeleton className="h-7 w-[66px]" />
        <Skeleton className="size-7" />
      </div>)}
    </LoadingRegion>
  </Section>
}

/** The Models page before its data: the same header, machine card and list it is about to show. */
export function ModelsSkeleton({ startingService = false }: { startingService?: boolean }) {
  return <div>
    <div className="mb-5 flex items-end justify-between gap-4"><h1 className={pageLayout.pageTitle}>Models</h1><Skeleton className="h-8 w-[132px]" /></div>
    {startingService && <HardwarePending startingService />}
    <MachineCardSkeleton />
    <YourModelsSkeleton />
  </div>
}
