import { ErrorNotice } from "./error-notice"
import { UsageActivity } from "./serving-usage-activity"
import { LoadingRegion, SkeletonLine } from "./page-skeletons"
import { pageLayout } from "./page-layout"
import { useState } from "react"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { Option } from "effect"
import { ServingModelId, type ServingUsageSnapshot } from "@magnitudedev/sdk"
import { useAgentClient, useLocalModels, formatLocalModelDisplayName } from "@magnitudedev/client-common"
import { GaugeIcon, TimerIcon, ArrowDownIcon, ArrowUpIcon, StackIcon, InfoIcon, ArrowsLeftRightIcon } from "@phosphor-icons/react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select"

import { ActionTooltip } from "./ui/tooltip"
import { readLanguage, translate } from "../i18n"

const tr = (key: Parameters<typeof translate>[1]) => translate(readLanguage(), key)

const number = (value: number) => value.toLocaleString()
const decimal = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 1 })
export function ServingUsage() {
  const [period, setPeriod] = useState<"Today" | "AllTime">("Today")
  const [model, setModel] = useState("")
  const client = useAgentClient()
  const result = useAtomValue(client.Models.GetServingUsage({ period, timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone, model: model ? Option.some(ServingModelId.make(model)) : Option.none() })).result
  const catalog = useLocalModels()
  const names = Result.isSuccess(catalog) ? new Map(catalog.value.models.map(model => [String(model.modelId), formatLocalModelDisplayName(model)])) : new Map<string, string>()
  const snapshot = Result.isSuccess(result) ? result.value : null
  return <section aria-label={tr("localUsage")} className="mt-6 space-y-6">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <Select value={model} onValueChange={value => setModel(value ?? "")}>
        <SelectTrigger aria-label={tr("usageModel")} className="w-[240px] max-w-full">
          <SelectValue>{model ? names.get(model) ?? model : tr("allModels")}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="">{tr("allModels")}</SelectItem>
          {snapshot?._tag === "Available" && snapshot.models.map(item => <SelectItem key={item.id} value={item.id}>{names.get(item.id) ?? item.id}</SelectItem>)}
          {model && (snapshot?._tag !== "Available" || !snapshot.models.some(item => item.id === model)) && <SelectItem value={model}>{names.get(model) ?? model}</SelectItem>}
        </SelectContent>
      </Select>
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-500">{tr("totals")}</span>
        <div className="flex rounded-lg bg-slate-100 p-1 dark:bg-slate-900" aria-label={tr("usagePeriod")}>
          {(["Today", "AllTime"] as const).map(value => <button key={value} aria-pressed={period === value} onClick={() => setPeriod(value)} className={`rounded-md px-3 py-1.5 text-sm ${period === value ? "bg-white text-blue-700 shadow-sm dark:bg-slate-750 dark:text-blue-300" : "text-slate-500"}`}>{value === "Today" ? tr("today") : tr("allTime")}</button>)}
        </div>
      </div>
    </div>
    {snapshot?._tag === "Available" ? <UsageActivity days={snapshot.dailyActivity} /> : Result.isInitial(result) ? <UsageActivity days={null} /> : null}
    <div className="space-y-6 border-t border-slate-200 pt-6 dark:border-slate-750">
      {snapshot?._tag === "Available" ? <UsageFigures usage={snapshot} /> : Result.isInitial(result) ? <LoadingRegion label={tr("loadingUsage")}><div className="space-y-6"><UsageFigures usage={null} /></div></LoadingRegion> : <ErrorNotice title={tr("couldNotLoadUsage")} description={tr("usageUnavailable")} />}
    </div>
  </section>
}
export function UsageFigures({ usage }: { usage: Extract<ServingUsageSnapshot, { _tag: "Available" }> | null }) {
  const missingCache = usage ? usage.requests - usage.cachedInputRequests : 0
  const cachePercentage = !usage || usage.inputTokens === 0 ? tr("noInputTokens") : usage.cachedInputRequests === 0 ? tr("cacheNotReported") : tr("cachePercentage").replace("{percent}", decimal(usage.cachedInputTokens / usage.inputTokens * 100)) + (missingCache > 0 ? ` ${tr("cacheNotReportedRequests").replace("{count}", number(missingCache))}` : "")
  const metrics = [
    { id: "Input tokens", label: tr("inputTokens"), value: usage ? number(usage.inputTokens) : null, icon: ArrowDownIcon, note: null },
    { id: "Cached input", label: tr("cachedInput"), value: !usage ? null : usage.cachedInputRequests === 0 && usage.requests > 0 ? "—" : number(usage.cachedInputTokens), icon: StackIcon, note: missingCache > 0 ? tr("cacheNotReportedForRequests").replace("{count}", number(missingCache)) : null },
    { id: "Output tokens", label: tr("outputTokens"), value: usage ? number(usage.outputTokens) : null, icon: ArrowUpIcon, note: null },
  ]
  return <>
    <div className={pageLayout.usageTokens}>{metrics.map(({ id, label, value, icon: Icon, note }) => <div key={id} className="min-w-0"><p className="flex items-center gap-2 text-sm text-slate-500"><Icon className="size-4" />{label}{id === "Cached input" && usage && <ActionTooltip label={cachePercentage} trigger={<button type="button" aria-label={cachePercentage} className="inline-flex rounded-sm text-slate-500 hover:text-slate-700 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-300"><InfoIcon aria-hidden="true" className="size-3.5" /></button>} />}</p><p className="mt-3 font-heading text-3xl tabular-nums" data-usage={id}>{value ?? <SkeletonLine className="h-9 text-3xl" width="100px" />}</p>{note && <p className="mt-1 text-xs text-slate-500">{note}</p>}</div>)}</div>
    <div className={`${pageLayout.usageTiming} border-t border-slate-200 pt-5 dark:border-slate-750`}>
      <div className="min-w-0"><p className="flex items-center gap-2 text-sm text-slate-500"><ArrowsLeftRightIcon className="size-4" />{tr("requests")}</p><p className="mt-4 text-2xl tabular-nums" data-usage="requests">{usage ? number(usage.requests) : <SkeletonLine className="h-8 text-2xl" width="100px" />}</p></div>
      <div className="min-w-0"><p className="flex items-center gap-2 text-sm text-slate-500"><GaugeIcon className="size-4" />{tr("generationSpeed")}</p><p className="mt-4 text-2xl tabular-nums" data-usage="speed">{!usage ? <SkeletonLine className="h-8 text-2xl" width="160px" /> : usage.tokensPerSecond === null ? "—" : `${decimal(usage.tokensPerSecond)} ${tr("tokensPerSecond")}`}</p></div>
      <div className="min-w-0"><p className="flex items-center gap-2 text-sm text-slate-500"><TimerIcon className="size-4" />{tr("timeToFirstToken")}</p><p className="mt-4 text-2xl tabular-nums" data-usage="ttft">{!usage ? <SkeletonLine className="h-8 text-2xl" width="120px" /> : usage.timeToFirstTokenMs === null ? "—" : usage.timeToFirstTokenMs < 1000 ? `${decimal(usage.timeToFirstTokenMs)} ${tr("milliseconds")}` : `${decimal(usage.timeToFirstTokenMs / 1000)} ${tr("secondsShort")}`}</p></div>
    </div>
  </>
}
