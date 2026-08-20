import { useSyncExternalStore, type ReactNode } from "react"
import { Option } from "effect"
import type { DisplayRootStatus } from "@magnitudedev/sdk"
import {
  displayRootStatusElapsedMs,
  formatElapsedMs,
  formatTokenCount,
  getTickSnapshot,
  subscribeNoop,
  subscribeTick,
  useStabilizedRootDetail,
  type LocalModelLoadActivity,
} from "@magnitudedev/client-common"
import { Progress } from "@/components/ui/progress"
import { Spinner } from "@/components/ui/spinner"

const activeTextClass =
  "animate-shimmer bg-linear-to-r from-slate-700 via-slate-400 to-slate-700 bg-[length:200%_100%] bg-clip-text text-transparent motion-reduce:animate-none motion-reduce:[background:none] motion-reduce:text-slate-800 dark:from-slate-400 dark:via-white dark:to-slate-400 dark:motion-reduce:text-slate-200"

function elapsed(status: Extract<DisplayRootStatus, { readonly _tag: "Working" }>): string {
  return formatElapsedMs(displayRootStatusElapsedMs(status, Date.now()))
}

function ActivityText({
  children,
  duration,
}: {
  readonly children: ReactNode
  readonly duration: string
}): ReactNode {
  return (
    <div className="flex min-h-6 items-baseline gap-3 font-sans text-[13px]">
      <span className={`${activeTextClass} font-medium`}>{children}</span>
      <span className="tabular-nums text-[12px] text-slate-500">{duration}</span>
    </div>
  )
}

function ModelLoadingActivity({
  activity,
  modelName,
}: {
  readonly activity: LocalModelLoadActivity
  readonly modelName: string | null
}): ReactNode {
  const residency = activity.residency
  if (residency._tag !== "Requested" && residency._tag !== "Loading") return null
  const percentage = residency._tag === "Loading"
    ? Option.match(residency.progress, {
        onNone: () => null,
        onSome: (progress) => Math.min(100, Math.max(0, Math.round(progress * 100))),
      })
    : null
  const label = modelName === null ? "Loading model" : `Loading ${modelName}`
  return (
    <div
      className="w-full max-w-[420px] py-0.5 font-sans"
      aria-live="polite"
      data-activity-kind="model-loading"
    >
      <div className={`${percentage === null ? "" : "mb-2"} flex items-center gap-2.5 text-[13px]`}>
        <Spinner className="size-3.5 shrink-0 text-blue-600 motion-reduce:animate-none dark:text-blue-500" />
        <span className="min-w-0 flex-1 truncate font-medium text-slate-800 dark:text-slate-200">
          {label}
        </span>
        {percentage !== null && (
          <span className="shrink-0 tabular-nums text-[12px] text-slate-500">
            {percentage}%
          </span>
        )}
      </div>
      {percentage !== null && (
        <Progress
          value={percentage}
          aria-label={`${label} ${percentage}%`}
          trackClassName="h-1 bg-slate-200 dark:bg-slate-800"
          indicatorClassName="bg-blue-600 transition-[width] duration-200 dark:bg-blue-500"
        />
      )}
    </div>
  )
}

function PrefillActivity({
  status,
  detail,
}: {
  readonly status: Extract<DisplayRootStatus, { readonly _tag: "Working" }>
  readonly detail: Extract<Extract<DisplayRootStatus, { readonly _tag: "Working" }>["detail"], { readonly _tag: "Prefill" }>
}): ReactNode {
  const total = Math.max(0, Math.floor(detail.totalTokens))
  const cached = Math.min(total, Math.max(0, Math.floor(detail.cachedTokens)))
  const completed = Math.min(total, Math.max(0, Math.floor(detail.completedTokens)))
  const effectiveTotal = Math.max(0, total - cached)
  const effectiveCompleted = Math.min(effectiveTotal, Math.max(0, completed - cached))

  return (
    <div
      className="flex min-h-6 flex-wrap items-baseline gap-x-3 gap-y-1 py-0.5 font-sans"
      aria-live="polite"
    >
      <span className={`${activeTextClass} font-medium text-[13px]`}>Preparing context</span>
      <span className="tabular-nums text-[12px] text-slate-500">{elapsed(status)}</span>
      <span className="tabular-nums text-[12px] text-slate-500">
        {formatTokenCount(effectiveCompleted)} / {formatTokenCount(effectiveTotal)} tokens
      </span>
      {cached > 0 && (
        <span className="tabular-nums text-[12px] text-slate-500">
          {formatTokenCount(cached)} cached
        </span>
      )}
    </div>
  )
}

export interface InlineWorkActivityProps {
  readonly rootStatus: DisplayRootStatus | null
  readonly modelLoadActivity: LocalModelLoadActivity | null
  readonly modelName: string | null
  readonly suppressGeneric?: boolean
}

export function InlineWorkActivity({
  rootStatus,
  modelLoadActivity,
  modelName,
  suppressGeneric = false,
}: InlineWorkActivityProps): ReactNode {
  const working = rootStatus?._tag === "Working"
  const tick = useSyncExternalStore(
    working ? subscribeTick : subscribeNoop,
    getTickSnapshot,
    getTickSnapshot,
  )
  void tick
  const detail = useStabilizedRootDetail(rootStatus)

  // Slot residency is app-global state and may outlive (or briefly lag) the
  // conversation turn that required it. It is conversation activity only
  // while the authoritative root actor is working.
  if (!working) return null
  if (modelLoadActivity?.residency._tag === "Requested"
    || modelLoadActivity?.residency._tag === "Loading") {
    return <ModelLoadingActivity activity={modelLoadActivity} modelName={modelName} />
  }
  if (suppressGeneric) return null
  if (detail?._tag === "Prefill") return <PrefillActivity status={rootStatus} detail={detail} />
  if (detail?._tag === "WaitingForModel") {
    return <ActivityText duration={elapsed(rootStatus)}>Waiting for model</ActivityText>
  }
  return <ActivityText duration={elapsed(rootStatus)}>Working</ActivityText>
}
